# test best B
/home/nemo/miniconda3/envs/nlp2021-hw2/bin/python /media/nemo/DATA/uni/nlp-hw2/hw2/stud/task_b.py train -a relu -n 3 -m roberta-base -f text -d 0.1 -lr 5e-5       
# test best C
/home/nemo/miniconda3/envs/nlp2021-hw2/bin/python /media/nemo/DATA/uni/nlp-hw2/hw2/stud/task_c.py train -a selu -n 3 -m roberta-base -d 0.1 -lr 5e-5      
# test best D
/home/nemo/miniconda3/envs/nlp2021-hw2/bin/python /media/nemo/DATA/uni/nlp-hw2/hw2/stud/task_d.py train -a tanh -n 2 -m roberta-base -f text -d 0.1 -lr 5e-5       








# end testing
echo 'shutting down in 3 minutes ...'
sleep 60
echo 'shutting down in 2 minutes ...'
sleep 60
echo 'shutting down in 1 minutes ...'
sleep 60
#sudo shutdown now