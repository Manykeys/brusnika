import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from torch import tensor
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import ollama

def parser():
    f = open('employees_info.json').read()
    data_table = json.loads(f)
    person_req = {}
    pr = {}
    collisions = defaultdict(int)
    count = 0
    for j in data_table:
        for q in data_table[j]:
            for w in data_table[j][q]:
                count += 1
                key = w + 'id=' + str(collisions[w])
                person_req[key] = data_table[j][q][w]['Функции']
                person_req[key].append(q)
                person_req[key].append(data_table[j][q][w]['Должность'])
                pr[key] = "".join(data_table[j][q][w]['Функции']) + data_table[j][q][w]['Должность'] + q
                collisions[w] += 1
    data_table = person_req
    return pr, data_table

def func1(s, pr , data_table):
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    embeddings2 = model.encode(list(pr.values()))
    dat = model.encode(s)
    arr = []
    for ind, value in enumerate(embeddings2):
        arr.append([data_table[list(data_table.keys())[ind]], util.pytorch_cos_sim(embeddings2[ind], dat), list(data_table.keys())[ind]])
    arr.sort(key=lambda x: x[1], reverse=True)
    return arr


def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Отправь мне запрос, и я выведу наиболее релевантных сотрудников.')

def echo(update: Update, context: CallbackContext) -> None:
    s = update.message.text
    print(s)
    pr, data_table = parser()
    persons = func1(s, pr, data_table)
    diff = tensor([[0.1044]])
    data = [{"functions": i[0][:-1], "name": i[2]} for i in persons if persons[0][1] - diff <= i[1]]
    
    data1 = data[:10]
    #print(prompt)
    stream = ollama.generate(
        model='llama3',
        prompt=f"Using this data: {data}. Select most suitable employees(name and id)(sort by relevance) and return answer in json format for request: {s}",
        options=
        {
            'temperature': 0.8,
        }
    )
    answer = str(stream["response"])
    #print(answer, type(answer))
    
    employees = json.loads(answer.split('```')[1].replace('json', ''))

    res = []
    employee_names_with_ids = [f'{employee["name"]}id={employee["id"]}' for employee in employees]

    for employee_name_with_id in employee_names_with_ids:
        res.append((data_table[employee_name_with_id], employee_name_with_id))
    counter = 0
    for i in res:
        formatted = f'{"".join(i[1])} \n✉️ e.fomicheva@brusnika.ru \nТел: +7 (965) 524 15 57 \nКомпания: ООО "Брусника" \nМесто работы: г. Екатеринбург, Гоголя, 18, 4 этаж \nДолжность: {i[0][-1]} \nФункции: {" ".join(i[0][:-1])} \nДата рождения: 30.07.1997'
        update.message.reply_text(formatted)
        counter += 1
        if counter == 3:
            break

def main() -> None:
    updater = Updater("7026382273:AAE1emD7ubHMqgRWJy6uJ-4qMEoc_xYomC8")
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
