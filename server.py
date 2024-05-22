import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from torch import tensor

def parser():
    f = open('employees_info.json').read()
    data_table = json.loads(f)
    #print(data_table)
    person_req = {}
    pr = {}
    collisions = defaultdict(int)
    #print(data_table.keys().__len__())
    count = 0
    for j in data_table:
        for q in data_table[j]:
            for w in data_table[j][q]:
                count += 1
                key = w + ' ' + str(collisions[w])
                person_req[key] = data_table[j][q][w]['Функции']
                person_req[key].append(q)
                person_req[key].append(data_table[j][q][w]['Должность'])
                pr[key] = "".join(data_table[j][q][w]['Функции']) + q + data_table[j][q][w]['Должность']
                collisions[w] += 1
    data_table = person_req
    #print(data_table)
    #print("Количество", count)
    #print(person_req['Иванов Петр 0'])
    return pr, data_table

def func1(s, pr , data_table):
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    embeddings2 = model.encode(list(pr.values()))
    dat = model.encode(s)
    arr = []
    for ind, value in enumerate(embeddings2):
        arr.append([data_table[list(data_table.keys())[ind]], util.pytorch_cos_sim(embeddings2[ind], dat), list(data_table.keys())[ind]])
        #print(data_table[ind], util.pytorch_cos_sim(embeddings[ind], dat))
    arr.sort(key=lambda x: x[1], reverse=True)
    return arr
    
def prepare_llama():
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    llm = Llama( 
        model_path=r"C:\Users\kopyr\Downloads\mistral-7b-instruct-v0.2.Q3_K_M.gguf",  # Download the model file first
        n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=124,         # The number of layers to offload to GPU, if you have GPU acceleration available
    )
    return llm

def generate_answer(llm, s, data):
    rrr = []
    for i in data:
        print(i)
        rrr.append(i[2] + " " + "".join(i[0]))
    answer = llm.create_chat_completion(
    messages = [
        {
            "role": "user",
            "content": "Выведи 3 сотрудников (вместе с id) которые больше всего релевантны к запросу:" + s + "\n"+ "\n".join(rrr)
        }
    ]
    )   
    return answer['choices'][0]['message']['content']

def main():
    pr, data_table = parser()
    print('Вводите')
    s = input()
    persons = func1(s, pr, data_table)
    diff = tensor([[0.1044]])
    print(persons[0][1])
    prompt = [i for i in persons if persons[0][1] - diff <= i[1]]
    prompt = prompt[:10]
    print(prompt)
    llm = prepare_llama()
    print(generate_answer(llm, s, prompt))

if __name__ == "__main__":
    main()
