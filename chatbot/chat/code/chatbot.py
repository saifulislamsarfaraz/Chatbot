from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch
import numpy as np

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

tokenizer_for_bert = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def bert_question_answer(question, passage, max_len=500):
    
  

    #Tokenize input question and passage 
    #Add special tokens - [CLS] and [SEP]
    input_ids = tokenizer_for_bert.encode (question, passage,  max_length= max_len, truncation=True)  
    """
    [101, 2054, 2003, 1996, 2171, 1997, 7858, 3149, 102, 3422, 3143, 2377, 9863, 1997, 3019, 2653, 6364, 1012, 
    2123, 1005, 1056, 5293, 2000, 2066, 1010, 3745, 1998, 4942, 29234, 2026, 3149, 1045, 2290, 6627, 2136, 102]
    """

    #Getting number of tokens in 1st sentence (question) and 2nd sentence (passage that contains answer)
    sep_index = input_ids.index(102) 
    len_question = sep_index + 1   
    len_passage = len(input_ids)- len_question  
    """
    8
    9
    27
    """
    
    #Need to separate question and passage
    #Segment ids will be 0 for question and 1 for passage
    segment_ids =  [0]*len_question + [1]*(len_passage)  
    """
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    """

    #Converting token ids to tokens
    tokens = tokenizer_for_bert.convert_ids_to_tokens(input_ids) 


    #Getting start and end scores for answer
    #Converting input arrays to torch tensors before passing to the model
    start_token_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]) )[0]
    end_token_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]) )[1]
    """
    tensor([[-5.9787, -3.0541, -7.7166, -5.9291, -6.8790, -7.2380, -1.8289, -8.1006,
         -5.9786, -3.9319, -5.6230, -4.1919, -7.2068, -6.7739, -2.3960, -5.9425,
         -5.6828, -8.7007, -4.2650, -8.0987, -8.0837, -7.1799, -7.7863, -5.1605,
         -8.2832, -5.1088, -8.1051, -5.3985, -6.7129, -1.4109, -3.2241,  1.5863,
         -4.9714, -4.1138, -5.9107, -5.9786]], grad_fn=<SqueezeBackward1>)
    tensor([[-2.1025, -2.9121, -5.9192, -6.7459, -6.4667, -5.6418, -1.4504, -3.1943,
         -2.1024, -5.7470, -6.3381, -5.8520, -3.4871, -6.7667, -5.4711, -3.9885,
         -1.2502, -4.0869, -6.4930, -6.3751, -6.1309, -6.9721, -7.5558, -6.4056,
         -6.7456, -5.0527, -7.3854, -7.0440, -4.3720, -3.8936, -2.1085, -5.8211,
         -2.0906, -2.2184,  1.4268, -2.1026]], grad_fn=<SqueezeBackward1>)
    """

    #Converting scores tensors to numpy arrays
    start_token_scores = start_token_scores.detach().numpy().flatten()
    end_token_scores = end_token_scores.detach().numpy().flatten()
    """
    [-5.978666  -3.0541189 -7.7166095 -5.929051  -6.878973  -7.238004
    -1.8289301 -8.10058   -5.9786286 -3.9319289 -5.6229596 -4.191908
    -7.20684   -6.773916  -2.3959794 -5.942456  -5.6827617 -8.700695
    -4.265001  -8.09874   -8.083673  -7.179875  -7.7863474 -5.16046
    -8.283156  -5.108819  -8.1051235 -5.3984528 -6.7128663 -1.4108785
    -3.2240815  1.5863497 -4.9714    -4.113782  -5.9107194 -5.9786243]

    [-2.1025064 -2.912148  -5.9192414 -6.745929  -6.466673  -5.641759
    -1.4504088 -3.1943028 -2.1024144 -5.747039  -6.3380575 -5.852047
    -3.487066  -6.7667046 -5.471078  -3.9884708 -1.2501552 -4.0868535
    -6.4929943 -6.375147  -6.130891  -6.972091  -7.5557766 -6.405638
    -6.7455807 -5.0527067 -7.3854156 -7.043977  -4.37199   -3.8935976
    -2.1084964 -5.8210607 -2.0906193 -2.2184045  1.4268283 -2.1025767]
    """

    #Getting start and end index of answer based on highest scores
    answer_start_index = np.argmax(start_token_scores)
    answer_end_index = np.argmax(end_token_scores)
    
    """
    31
    34
    """

    #Getting scores for start and end token of the answer
    start_token_score = np.round(start_token_scores[answer_start_index], 4)
    end_token_score = np.round(end_token_scores[answer_end_index], 4)
    """
    1.59
    1.43
    """

    #Combining subwords starting with ## and get full words in output. 
    #It is because tokenizer breaks words which are not in its vocab.
    answer = tokens[answer_start_index] 
    for i in range(answer_start_index + 1, answer_end_index + 1):
        if tokens[i][0:2] == '##':  
            answer += tokens[i][2:] 
        else:
            answer += ' ' + tokens[i]  

    # If the answer didn't find in the passage
    if ( answer_start_index == 0) or (start_token_score < 0 ) or  (answer == '[SEP]') or ( answer_end_index <  answer_start_index):
        answer = "Sorry!, I could not find an answer in the passage."
    
    return (answer_start_index, answer_end_index, start_token_score, end_token_score,  answer)


# Let me define one passage
passage = """ In short, Quantum is the science of well-being. It is the science that enables us to transform our desires into reality and change ourselves the way we want to. It gives us the privilege to say 'I am doing great', in every moment of our lives, and it is possible for anyone to achieve this state of being. Three decades of success of those practicing the Quantum Method have proven this to be true. 
This life is pretty simple. You don't need to be an expert or a specialist to become happy in life. All you need is the right attitude.When we gain the right attitude, our life can flow with the ease and spontaneity of a flowing spring. It helps us utilize the limitless power of our mind to create a new reality, to attain greater success.The truth is, all our problems are rooted in our minds. When our mental energy gets tangled up, diseases develop. Possibilities turn into problems, and hope is replaced by frustration.Scientists say 75% of the diseases are psychosomatic.That is, even though the diseases affect the body, they are rooted in the mind. That is why we strongly claim: “Come, meditate with us and untangle the knots of your mind. You will definitely get better. ”
The same rule applies in the case of failures in life.It is a scientific process that transforms a person from within. This is the reality of this new millennium. This is the new truth."""

print (f'Length of the passage: {len(passage.split())} words')

question1 = input("What you wnated to know: ") 
print ('\nQuestion 1:\n', question1)
_, _ , _ , _, ans  = bert_question_answer(question1, passage)
print('\nAnswer from BERT: ', ans ,  '\n')
