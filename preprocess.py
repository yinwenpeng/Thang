

def preprocess_thang_files():
    train_file_1=open('/mounts/Users/cisintern/thangvu/software/Kaldi/kaldi-trunk/egs/wsj/s5/data/local/data/test_dev93.txt')
    #train_file_2=open('/mounts/Users/cisintern/thangvu/software/Kaldi/kaldi-trunk/egs/wsj/s5/data/local/data/test_eval93.txt')
    
    write_train=open('/mounts/data/proj/wenpeng/Thang/dev_dev93.txt','w')
    max_sent_length=0
    min_sent_length=500
    
    for line_1 in train_file_1:
        splits=line_1.strip().split()
        if len(splits)-1>max_sent_length:
            max_sent_length=len(splits)-1
        if len(splits)-1<min_sent_length:
            min_sent_length=len(splits)-1
        new_sent=''
        for i in range(1,len(splits)):
            new_sent+=splits[i].lower()+' '
        write_train.write(new_sent.strip()+'\n')
    '''   
    for line_2 in train_file_2:
        splits=line_2.strip().split()
        if len(splits)-1>max_sent_length:
            max_sent_length=len(splits)-1
        if len(splits)-1<min_sent_length:
            min_sent_length=len(splits)-1
        new_sent=''
        for i in range(1,len(splits)):
            new_sent+=splits[i].lower()
        write_train.write(new_sent+'\n')   
    '''
    write_train.close()
    train_file_1.close()
    #train_file_2.close()    
    print 'max:', max_sent_length
    print 'min:', min_sent_length
        
        
if __name__ == '__main__':
    preprocess_thang_files()