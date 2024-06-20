def solution(babbling):
    answer = 0
    speak = ["aya", "ye", "woo", "ma"]
    for sentence in babbling:
        for word in speak:
            if word in sentence:
                sentence = sentence.replace(word,'')
                if sentence == '':
                    answer+=1
    return answer
babbling = ["ayaye", "uuuma", "ye", "yemawoo", "ayaa"]
answer = solution(babbling)
print(answer)