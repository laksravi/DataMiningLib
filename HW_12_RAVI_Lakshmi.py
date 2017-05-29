import numpy as np
import  sys

def count_trigram(content):
    '''
    store the combinations ina 3-D array and end of index as 26th index of 3rd array
    :param content:
    :return:
    '''
    trigram_count = np.zeros((26,26,27))
    for word in content:
        len_word = len(word)
        first_index, second_index, third_index = -1,-1,-1
        for i in range(len_word-2):
            first_index = ord(word[i])-ord('a')
            second_index = ord(word[i+1]) - ord('a')
            third_index = ord(word[i + 2]) - ord('a')
            if first_index < 26 and second_index < 26 and third_index < 26:
                trigram_count[first_index][second_index][third_index]+=1
        #append the end of word
        if first_index < 26 and second_index < 26:
            trigram_count[first_index][second_index][26]+=1

    return trigram_count

def count_bigram(content):
    '''
    count the combinations of words in a 2-D array
    :param content:
    :return:
    '''
    bigram_count = np.zeros((26,26))
    for word in content:
        len_word = len(word)
        for i in range(len_word-1):
            first_index = ord(word[i])-ord('a')
            second_index = ord(word[i+1]) - ord('a')
            if first_index < 26 and second_index < 26:
                bigram_count[first_index][second_index]+=1

    return bigram_count

def count_start(content):
    '''
    Count the beginning character count in the words given
    :param content:
    :return:
    '''
    start_letter_count = np.zeros(26);
    for word in content:
        start_letter = word[0]
        start_letter_index = ord(start_letter) - ord('a')
        if start_letter_index < 26:
            start_letter_count[start_letter_index]+=1
    return start_letter_count

def count_end(content):
    '''
        Count the end character count in the words given
        :param content:
        :return:
        '''
    end_letter_count = np.zeros(26);
    for word in content:
        end_letter = word[len(word)-1]
        end_letter_index = ord(end_letter) - ord('a')
        if end_letter_index < 26:
            end_letter_count[end_letter_index] += 1
    return end_letter_count


def read_file(words_file_name):
    content=[]
    with open(words_file_name) as f:
        for line in f:
            line=line.replace('\n', '')
            line=line.replace('\'', '')
            line = line.replace('ã³', '')
            line=line.lower()
            string=[]
            for char in line:
                if char.isalnum():
                    string.append(char)
            content.append(string)
    f.close()
    return content

def trigram_analysis(trigram_count):
    e_index = ord('e')- ord('a')
    d_index = ord('d')- ord('a')
    r_index= ord('r') - ord('a')
    #er occurs in the end or ed occurs in the end more - count comparison
    if trigram_count[e_index][r_index][26] > trigram_count[e_index][d_index][26]:
        print("er occurs more than ed - as suffix", trigram_count[e_index][r_index][26])
    else:
        print("ed occurs more than er - as suffix", trigram_count[e_index][d_index][26])

    #most common combination ant or ent ?
    a_index = 0
    n_index = ord('n') - ord('a')
    t_index = ord('t') - ord('a')
    if trigram_count[a_index][n_index][t_index] > trigram_count[e_index][n_index][t_index]:
        print(" ant occurs more than ent", trigram_count[a_index][n_index][t_index] )
    else:
        print("  ent occurs more than ant", trigram_count[e_index][n_index][t_index])

    #most common combination tio or ion
    t_index = ord('t') - ord('a')
    i_index = ord('i') - ord('a')
    o_index = ord('o') - ord('a')
    if trigram_count[t_index][i_index][o_index] > trigram_count[i_index][o_index][n_index]:
        print("tio occurs more than ion", trigram_count[t_index][i_index][o_index])
    else:
        print("ion occurs more than tio", trigram_count[i_index][o_index][n_index])

    #
    if trigram[e_index][n_index][e_index] > trigram[i_index][n_index][e_index]:
        print("ene occurs more than ine", trigram[e_index][n_index][e_index])
    else:
        print("ine occurs more than ene", trigram[i_index][n_index][e_index])

    q_index = ord('q') - ord('a')
    u_index = ord('u') - ord('a')
    max_next_index = chr(np.argmax(trigram_count[q_index, u_index, :])+ord('a'))
    print(" character occuring next to qu is", max_next_index, "occuring", np.max(trigram_count[q_index][u_index]))

def bigram_analysis(bigram_count):
    '''
        returns the analysis of bigram array
    :param bigram_count:
    :return:
    '''
    #double n's or double s'
    n_index = ord('n') - ord('a')
    s_index = ord('s') - ord('a')
    if bigram_count[n_index][n_index] > bigram_count[s_index][s_index]:
        print("Double -n's occur many times", bigram_count[n_index][n_index])
    else:
        print("Double- S's occur many times", bigram_count[s_index][s_index])

    #letter most likely to be coming after t is
    t_index = ord('t') - ord('a')
    letter_index = np.argmax(bigram[t_index,:])
    letter_count = np.max(bigram[t_index,:])
    letter_after_t = chr(letter_index+ord('a'))
    print("The vowel coming after t most of the times is..", letter_after_t, "occuring", letter_count)

    # letter most likely to be coming after u is
    u_index = ord('u') - ord('a')
    letter_index = np.argmax(bigram[u_index,:])
    count_after_u = np.max(bigram[u_index,:])

    # most common letter after u
    letter_after_u = chr(letter_index + ord('a'))
    print("The letter coming after u most of the times is..", letter_after_u, "occuring..", count_after_u)

    #second common occurance after 'u' next to 'q'
    q_index = ord('q')- ord('a')
    q_neighbor_index =np.argsort(bigram_count[q_index, :])
    print("Second common neighbor of q is", chr(q_neighbor_index[24]+ord('a')), "occuring", bigram[q_index][q_neighbor_index[24]])
    for i in range(26):
        max_index = np.argmax(bigram_count[i,:])
        print(chr(i+ord('a')), " 's next occuring character is", chr(max_index+ord('a')) )

def mostcommon_character(count_array, type):
    '''
    find the most common character from the given count-array
    :param count_array:
    :param type:
    :return:
    '''
    most_occuring = chr(np.argmax(count_array)+ord('a'))
    times = np.max(count_array)
    print("Most occuring", type, " character is...", most_occuring,"... occuring", times, " times")


#execution of analysis
#read from file, count bigram and trigram
content = sys.argv[1]
bigram = count_bigram(content)
trigram = count_trigram(content)

bigram_analysis(bigram)
trigram_analysis(trigram)
print("Find the most occuring Start and end character")
start_count=count_start(content)
end_count = count_end(content)
print(mostcommon_character(start_count, "Start"))
print(mostcommon_character(end_count, "end"))
print("\n\n")