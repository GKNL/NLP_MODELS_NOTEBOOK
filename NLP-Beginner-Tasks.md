# NLP-Beginner����Ȼ���Դ���������ϰ

�¼��뱾ʵ���ҵ�ͬѧ���밴Ҫ�����������ϰ�����ύ���档

*�����ÿ����ϰ���report�ϴ���QQȺ�еĹ����ļ����еġ�Reports of nlp-beginner��Ŀ¼���ļ�������ʽΪ��task 1+��������*

�ο���

1. [���ѧϰ����ָ��](https://github.com/nndl/nndl.github.io/blob/master/md/DeepGuide.md)
2. ��[�����������ѧϰ](https://nndl.github.io/)�� 
3. ������google





### ����һ�����ڻ���ѧϰ���ı�����

ʵ�ֻ���logistic/softmax regression���ı�����

1. �ο�
   1. [�ı�����](�ı�����.md)
   2. ��[�����������ѧϰ](https://nndl.github.io/)�� ��2/3��
2. ���ݼ���[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
3. ʵ��Ҫ��NumPy
4. ��Ҫ�˽��֪ʶ�㣺

   1. �ı�������ʾ��Bag-of-Word��N-gram
   2. ��������logistic/softmax  regression����ʧ��������������ݶ��½�������ѡ��
   3. ���ݼ���ѵ����/��֤��/���Լ��Ļ���
5. ʵ�飺
   1. ������ͬ����������ʧ������ѧϰ�ʶ����շ������ܵ�Ӱ��
   2. shuffle ��batch��mini-batch 
6. ʱ�䣺����

### ��������������ѧϰ���ı�����

��ϤPytorch����Pytorch��д������һ����ʵ��CNN��RNN���ı����ࣻ

1. �ο�

   1. https://pytorch.org/
   2. Convolutional Neural Networks for Sentence Classification <https://arxiv.org/abs/1408.5882>
   3. <https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/>
2. word embedding �ķ�ʽ��ʼ��
1. ���embedding�ĳ�ʼ����ʽ
  2. ��glove Ԥѵ����embedding���г�ʼ�� https://nlp.stanford.edu/projects/glove/
3. ֪ʶ�㣺

   1. CNN/RNN��������ȡ
   2. ��Ƕ��
   3. Dropout
4. ʱ�䣺����

### ������������ע�������Ƶ��ı�ƥ��

�������������жϣ��ж�����֮��Ĺ�ϵ���ο�[ESIM]( https://arxiv.org/pdf/1609.06038v3.pdf)������ֻ��LSTM������Tree-LSTM������˫���ע��������ʵ�֡�

1. �ο�
   1. ��[�����������ѧϰ](https://nndl.github.io/)�� ��7��
   2. Reasoning about Entailment with Neural Attention <https://arxiv.org/pdf/1509.06664v1.pdf>
   3. Enhanced LSTM for Natural Language Inference <https://arxiv.org/pdf/1609.06038v3.pdf>
2. ���ݼ���https://nlp.stanford.edu/projects/snli/
3. ʵ��Ҫ��Pytorch
4. ֪ʶ�㣺
   1. ע��������
   2. token2token attetnion
5. ʱ�䣺����


### �����ģ�����LSTM+CRF�����б�ע

��LSTM+CRF��ѵ�����б�עģ�ͣ���Named Entity RecognitionΪ����

1. �ο�
   1. ��[�����������ѧϰ](https://nndl.github.io/)�� ��6��11��
   2. https://arxiv.org/pdf/1603.01354.pdf
   3. https://arxiv.org/pdf/1603.01360.pdf
2. ���ݼ���CONLL 2003��https://www.clips.uantwerpen.be/conll2003/ner/
3. ʵ��Ҫ��Pytorch
4. ֪ʶ�㣺
   1. ����ָ�꣺precision��recall��F1
   2. ����ͼģ�͡�CRF
5. ʱ�䣺����

### �����壺���������������ģ��

��LSTM��GRU��ѵ���ַ���������ģ�ͣ����������

1. �ο�
   1. ��[�����������ѧϰ](https://nndl.github.io/)�� ��6��15��
2. ���ݼ���poetryFromTang.txt
3. ʵ��Ҫ��Pytorch
4. ֪ʶ�㣺
   1. ����ģ�ͣ�����ȵ�
   2. �ı�����
5. ʱ�䣺����