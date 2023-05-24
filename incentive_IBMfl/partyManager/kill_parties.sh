for i in {8000..8500}
do
   fuser -k $i/tcp
done
