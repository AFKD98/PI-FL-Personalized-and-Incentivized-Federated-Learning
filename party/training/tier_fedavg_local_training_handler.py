"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
import logging

from ibmfl.exceptions import LocalTrainingException, \
    ModelUpdateException
from ibmfl.model.model_update import ModelUpdate
import numpy as np
from scipy.spatial import distance
from ibmfl.party.training.local_training_handler import LocalTrainingHandler
import copy

logger = logging.getLogger(__name__)

class TierFedAvgLocalTrainingHandler(LocalTrainingHandler):

    def __init__(self, fl_models, fl_model, tiers, data_handler, hyperparams=None, evidencia=None, **kwargs):
        """
        Initialize LocalTrainingHandler with fl_model, data_handler

        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param evidencia: evidencia to use
        :type evidencia: `evidencia.EvidenceRecorder`
        :param kwargs: Additional arguments to initialize a local training \
        handler, e.g., a crypto library object to help with encryption and \
        decryption.
        :type kwargs: `dict`
        :return None
        """
        self.fl_models = fl_models
        self.fl_model = fl_model
        self.tiers = tiers
        self.data_handler = data_handler
        self.hyperparams = hyperparams
        self.evidencia = evidencia
        self.personalized_acc = 0.0
        

        self.metrics_recorder = None
        self.n_completed_trains = 0
        self.n_completed_evals = 0

        if self.evidencia:
            from ibmfl.evidencia.util.hashing import hash_np_array, \
                hash_model_update


    def set_metrics_recorder_obj(self, metrics_recorder):
        """
        Set metrics instance variable to the input parameter. We do this because the \
        party_protocol_handler tells the local_training_handler which metrics object to use; the \
        local_training_handler can be constructed somewhere else, so we don't want to force the \
        metrics object to necessarily exist at that time.

        :param metrics_recorder: Metrics-recording object (probably empty at time this is called)
        :type metrics_recorder: `MetricsRecorder`
        """
        self.metrics_recorder = metrics_recorder

    def update_model_by_tier(self, model_updates):
        """
        Update local model with model updates received from FusionHandler

        :param model_update: ModelUpdate
        :type model_update: `ModelUpdate`
        """
        try:
            if model_updates is not None:
                for i in range(len(self.fl_models)):
                    self.fl_models[i].update_model(model_updates[i])
                logger.info('Local model updated.')
            else:
                logger.info('No model update was provided.')
        except Exception as ex:
            raise LocalTrainingException('No query information is provided. '+ str(ex))

    def get_train_metrics_pre(self):
        """
        Call the post-train metrics hook. This hook runs immediately before the training starts at
        each party during the routine corresponding to the TRAIN command.

        :param: None
        :return: None
        """
        if self.metrics_recorder:
            try:
                # TODO: find sound way of determining if we really want to do a pre_train (i.e.
                # "synced model") eval
                if self.metrics_recorder.compute_pre_train_eval and self.get_n_completed_trains() > 0:
                    ret = self.data_handler.get_data()
                    if ret is not None:
                        (_), test_dataset = ret
                    else:
                        test_dataset = None
                    pre_eval_results = self.fl_model.evaluate(test_dataset)
                else:
                    pre_eval_results = None
                # collect metrics specific to the model class, that the user may customize
                additional_metrics = self.fl_model.get_custom_metrics_pre()
                self.metrics_recorder.pre_train_hook(pre_eval_results, additional_metrics)
            except Exception as e:
                logger.exception(str(e))
                raise LocalTrainingException(
                    'Error occurred while running pre-train hooks')

    def get_train_metrics_post(self):
        """
        Call the post-train metrics hook. This hook runs immediately after the training finishes at
        each party during the routine corresponding to the TRAIN command.

        :param: None
        :return: None
        """
        if self.metrics_recorder:
            try:
                train_result = self.fl_model.get_train_result()
                # TODO: find sound way of determining if we really want to do a post_train (i.e.
                # "locally-trained") eval
                if self.metrics_recorder.compute_post_train_eval:
                    ret = self.data_handler.get_data()
                    if ret is not None:
                        (_), test_dataset = ret
                    else:
                        test_dataset = None
                    post_eval_results = self.fl_model.evaluate(test_dataset)
                else:
                    post_eval_results = None
                additional_metrics = self.fl_model.get_custom_metrics_post()
                self.metrics_recorder.post_train_hook(train_result, post_eval_results, additional_metrics)
            except Exception as e:
                logger.exception(str(e))
                raise LocalTrainingException(
                    'Error occurred while running post-train hooks')

    def train_by_tier(self, fit_params=None):
        """
        Train locally using fl_model. At the end of training, a
        model_update with the new model information is generated and
        send through the connection.

        :param fit_params: (optional) Query instruction from aggregator
        :type fit_params: `dict`
        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        try:
            train_data, (_) = self.data_handler.get_data()
            _train_count = train_data[0].shape[0]


            if self.evidencia:
                self.evidencia.add_claim("training_data_hash", "'{}'".format(hash_np_array(train_data[0])))
                self.evidencia.add_claim("training_data_labels_hash", "'{}'".format(hash_np_array(train_data[1])))
                self.evidencia.add_claim("training_data_size", str(train_data[0].shape[0]))
                self.evidencia.add_claim("training_data_labels_number",
                                                    str(np.unique(train_data[1], axis=0).shape[0]))
                # TODO labels are hardcoded
                labels_text = ['Verkehr & Mobilität', 'Städtebau & Stadtraum', 'Sonstiges',
                                'Grün & Erholung', 'Soziales & Kultur',
                                'Wohnen & Arbeiten',
                                'Sport & Freizeit', 'Klima & Umweltschutz']
                self.evidencia.add_claim("labels_list", "'{}'".format(str(labels_text).replace('\'', '"')))
                # also log number of training instances per label
                (labels, counts) = np.unique(np.argmax(train_data[1], axis=1), return_counts=True)

                for idx, _ in np.ndenumerate(labels):
                    self.evidencia.add_claim("training_data_count_per_label",
                                                            '{}, {}'.format(labels[idx], counts[idx]))

            model_updates = fit_params.get('model_updates')
            self.update_model_by_tier(model_updates)
            # self.personalized_acc = self.update_personalized_model()


            if self.evidencia:
                self.evidencia.add_claim("received_model_update", "'\"{}\"'".format(
                hash_model_update(self.fl_model.get_model_update())))

            self.get_train_metrics_pre()

            logger.info('Local training started...')
            
            #[FARAZ] Training models for tiers which client belongs to
            tier_to_train = int(fit_params.get('hyperparams').get('tier'))
            logger.info('Training tier: {}'.format(tier_to_train))
            # logger.info("[FARAZ] Training model {}".format(self.fl_models[tier_to_train].get_model_update()))
            self.fl_models[tier_to_train].fit_model(train_data, fit_params, local_params=self.hyperparams)

            update = self.fl_models[tier_to_train].get_model_update()
            update.add('train_counts', _train_count)
            
            logger.info('Local training done, generating model update...')

            
            if self.evidencia:
                self.evidencia.add_claim("sent_model_update", "'\"{}\"'".format(hash_model_update(update)))

            # logger.info("[FARAZ] Model update {}".format(update.get('weights')[-1]))
            
            self.get_train_metrics_post()

            return update
        except Exception as e:
            logger.exception(str(e))
            raise LocalTrainingException('Error occurred while training')
    
    def get_model_logits(self, model):
        '''Returns the softmax layer logits of the model'''
        if isinstance(model, ModelUpdate):
            model = model.get('weights')

        return model[-1]
    
    def get_cosine_differences(self, tier_global_model_logits, client_model_logits, tier_id):
        # logger.info('[FARAZ] tier_global_model_logits: {}'.format(tier_global_model_logits))
        # logger.info('[FARAZ] client_model_logits: {}'.format(client_model_logits))
        return (distance.cosine(client_model_logits, tier_global_model_logits), tier_id)
        
    def update_personalized_model(self):
        try:
            if self.fl_models:
                evaluation_result = self.eval_model_by_tier()
                logger.info('Evaluation result: {}'.format(evaluation_result))
                
                results = [accuracy['accuracy_score'] for accuracy in (list(evaluation_result.values()))]
                # max_accurate_tier = np.argmax(results)
                # fl_model_weights = copy.deepcopy(np.array(self.fl_model.get_weights(to_numpy=True), dtype=object))*0.0
                fl_model_weights = np.zeros_like(np.array(self.fl_model.get_weights(to_numpy=True), dtype=object))
                
                _lambda = 0.01
                cutoff = 0.1
                # logger.info('results: {}'.format(results))
                # results[np.isnan(results)] = 0.0
                # create a boolean mask for the non-NaN values
                # logger.info('[FARAZ] Before results: {}'.format(results))
                # results = np.array(results)
                # # create a boolean mask for the non-NaN values
                # mask = ~np.isnan(results)
                # # select only the non-NaN values using the mask
                # results= results[mask]
                # # logger.info('results: {}'.format(results))
                # result_sorted_ids = np.argsort(results, axis=0)[::-1]
                # logger.info('[FARAZ] After results: {}'.format(result_sorted_ids))
                # for tier_id in range(0,max(1,int(len(result_sorted_ids)*cutoff))):
                #     tier = result_sorted_ids[tier_id]
                #     logger.info('importance weight of tier {}: {}'.format(tier, (results[tier]/sum(results))))
                #     fl_model_weights+= np.multiply((results[tier]/sum(results)),(np.array(self.fl_models[tier].get_weights(to_numpy=True), dtype=object)))
       
                for tier in range(0, self.tiers):
                    logger.info('importance weight of tier {}: {}'.format(tier, (results[tier]/sum(results))))
                    fl_model_weights+= np.multiply((results[tier]/sum(results)),(np.array(self.fl_models[tier].get_weights(to_numpy=True), dtype=object)))
                    # logger.info('Results: {}'.format(results))
                    # logger.info('self.fl_models[tier].get_weights(): {}'.format(type(self.fl_models[tier].get_model_update().get('weights'))))
                    # fl_model_weights+= _lambda*(results[tier]/sum(results))*np.square(abs(fl_model_weights - copy.deepcopy((np.array(self.fl_models[tier].get_weights(to_numpy=True), dtype=object)))))
                    # fl_model_weights+= (results[tier]/sum(results))*((np.array(self.fl_models[tier].get_weights(to_numpy=True), dtype=object)))
                    # logger.info('fl_model_weights: {}'.format(fl_model_weights[-1:]))
                    
                #check if personalized model accuracy improved otherwise keep the previous model
                # temp_personalized_update = self.fl_model.get_model_update()
                
                # personalized_update = {'weights': fl_model_weights}
                # self.fl_model.update_model(ModelUpdate(**personalized_update))
                
                # personalized_evaluation = self.eval_model()
                # if self.personalized_acc>personalized_evaluation['accuracy_score']:
                #     self.fl_model.update_model(temp_personalized_update)
                #     logger.info('[FARAZ] evaluation personalized model:' + str(personalized_evaluation))
                #     return self.personalized_acc

                # logger.info('[FARAZ] evaluation personalized model:' + str(personalized_evaluation))
                # return personalized_evaluation['accuracy_score']
            
                personalized_update = {'weights': fl_model_weights}
                self.fl_model.update_model(ModelUpdate(**personalized_update))
                personalized_evaluation = self.eval_model()
                logger.info('[FARAZ] evaluation personalized model:' + str(personalized_evaluation))
                return personalized_evaluation['accuracy_score']
            
        except Exception as e:
            logger.exception(str(e))
            raise LocalTrainingException('Error occurred while updating personalized model')

    def get_tier_model_accuracies(self, payload):
        """
        Send tier preferences to the aggregator

        :param payload: (optional) Query instruction from aggregator
        :type payload: `dict`
        :return: None
        """
        try:
            model_updates = payload['model_updates']
            
            self.update_model_by_tier(model_updates)
            accuracies = self.eval_model_by_tier(eval_dataset=self.data_handler.get_val_data())
            accuracies = [accuracy['accuracy_score'] for accuracy in (list(accuracies.values()))]
            logger.info('[FARAZ] tier model accuracies: {}'.format(accuracies))
            return accuracies
        except:
            logger.exception('Error occurred while sending tier model accuracies to the aggregator')
            return 0
                   
    def send_tier_preferences(self, payload):
        """
        Send tier preferences to the aggregator

        :param payload: (optional) Query instruction from aggregator
        :type payload: `dict`
        :return: None
        """
        try:
            # if payload is not None:
            # data = payload['tier_model_logits']
            # cosine_differences = []
            # for i in range(len(data)):
            #     client_model_logits = self.get_model_logits(self.fl_model.get_model_update())
            #     cosine_differences.append(self.get_cosine_differences(data[i], client_model_logits, i))
            # # logger.info('[FARAZ] payload: {}'.format(data))
            # # logger.info('[FARAZ] payload type: {}'.format(type(data)))
            # logger.info('[FARAZ] cosine differences: {}'.format(cosine_differences))
            # logger.info('Sending tier preferences to the aggregator...')
            # sorted_by_second = sorted(cosine_differences, key=lambda tup: tup[0])
            # logger.info('[FARAZ] choice of tier: {}'.format(sorted_by_second[0][1]))
            # return str(sorted_by_second[0][1])
            
            # if payload is not None:
            data = payload['model_updates']
            
            # for i in range(len(data)):
            #     client_model_logits = self.get_model_logits(self.fl_model.get_model_update())
            #     cosine_differences.append(self.get_cosine_differences(data[i], client_model_logits, i))
            # logger.info('[FARAZ] payload: {}'.format(data))
            # logger.info('[FARAZ] payload type: {}'.format(type(data)))
            # logger.info('[FARAZ] cosine differences: {}'.format(cosine_differences))
            sorted_by_second = []
            self.update_model_by_tier(data)
            accuracies = self.eval_model_by_tier(eval_dataset=self.data_handler.get_val_data())
            accuracies = [accuracy['accuracy_score'] for accuracy in (list(accuracies.values()))]
            logger.info('Sending tier preferences to the aggregator...')
            
            # for i in range(len(accuracies)):
            #     if accuracies[i] > 0.9:
            #         sorted_by_second.append(str(i))
            # if len(sorted_by_second) == 0:
            #     sorted_by_second.append(str(np.argmax(accuracies)))
            sorted_by_second.append(str(np.argmax(accuracies)))
            logger.info('[FARAZ] choice of tier: {}'.format(sorted_by_second))
            return sorted_by_second
        
        except:
            logger.exception('Error occurred while sending tier preferences to the aggregator')
            return 0
        
    def save_model_by_tier(self, payload=None):
        """
        Save the local model.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Status of save model request
        :rtype: `boolean`
        """
        logger.info()
        status = False
        try:
            for fl_model in self.fl_models:
                self.fl_model.save_model()
            status = True
        except Exception as ex:
            logger.error("Error occurred while saving local model")
            logger.exception(ex)

        return status

    def get_update_metrics_pre(self):
        """
        Call the pre-update metrics hook. This hook runs before the model update from the SYNC
        command, but after the SYNC command instruction has been received.

        :param: None
        :return: None
        """
        if self.metrics_recorder:
            try:
                self.metrics_recorder.pre_update_hook()
            except Exception as e:
                logger.exception(str(e))
                raise LocalTrainingException(
                    'Error occurred while running pre-update hooks')

    def get_update_metrics_post(self):
        """
        Call the post-update metrics hook. This hook runs after the model update from the SYNC
        command, but still during the routine corresponding to that SYNC.

        :param: None
        :return: None
        """
        if self.metrics_recorder:
            try:
                self.metrics_recorder.post_update_hook()
            except Exception as e:
                logger.exception(str(e))
                raise LocalTrainingException(
                    'Error occurred while running post-update hooks')

    def sync_model_impl_by_tier(self, payload=None):
        """
        Update the local model with global ModelUpdate received
        from the Aggregator. This function is meant to be 
        overridden in base classes as opposed to sync_model, which
        contains boilerplate for exception handling and metrics.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Status of sync model request
        :rtype: `boolean`
        """
        status = False
        model_updates = payload['model_updates']
        for i in range(0, len(self.fl_models)):
            status = self.fl_models[i].update_model(model_updates[i])
        return status
    
    def sync_model_by_tier(self, payload=None):
        """
        Update the local model with global ModelUpdate received
        from the Aggregator.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Status of sync model request
        :rtype: `boolean`
        """
        
        status = False
        if payload is None or 'model_updates' not in payload:
            raise ModelUpdateException(
                "Invalid Model update request aggregator")

        self.get_update_metrics_pre()

        try:
            status = self.sync_model_impl_by_tier(payload)
        except Exception as ex:
            logger.error("Exception occurred while sync model")
            logger.exception(ex)

        self.get_update_metrics_post()

        return status
    
    def get_personalized_model_accuracy(self, payload=None):
        '''
        returns accuracy of personalized model
        
        returns: accuracy
        type: float
        '''
        self.personalized_acc = self.update_personalized_model()
        return self.personalized_acc
    
    def eval_model(self, payload=None):
        """
        Evaluate the local model based on the local test data.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Dictionary of evaluation results
        :rtype: `dict`
        """

        (_), test_dataset = self.data_handler.get_data()
        evaluations = []
        try:
            
            evaluations = self.fl_model.evaluate(test_dataset)

        except Exception as ex:
            logger.error("Expecting the test dataset to be of type tuple. "
                         "However, test dataset is of type "
                         + str(type(test_dataset)))
            logger.exception(ex)

        return evaluations
    
    def eval_model_by_tier(self, eval_dataset=None):
        """
        Evaluate the local model based on the local test data.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Dictionary of evaluation results
        :rtype: `dict`
        """
        if eval_dataset is None:
            _, test_dataset = self.data_handler.get_data()
            eval_dataset = test_dataset
            
        evaluations = dict()
        try:
            i = 0
            for fl_model in self.fl_models:
                evaluations[str(i)] = fl_model.evaluate(eval_dataset)
                i=i+1
            logger.info('[FARAZ] evaluations:' + str(evaluations))

        except Exception as ex:
            logger.error("Expecting the test dataset to be of type tuple. "
                         "However, test dataset is of type "
                         + str(type(eval_dataset)))
            logger.exception(ex)

        return evaluations

    def get_n_completed_trains(self):
        """
        Return the number of completed executions of the TRAIN command at the party side

        :param: None
        :return: Number indicating how many TRAINs have been completed
        :rtype: `int`
        """
        return self.n_completed_trains

    def get_n_completed_evals(self):
        """
        Return the number of completed executions of the EVAL command at the party side

        :param: None
        :return: Number indicating how many EVALs have been completed
        :rtype: `int`
        """
        return self.n_completed_evals
