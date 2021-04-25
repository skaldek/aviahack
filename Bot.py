import json
import os
import re

import _jsonnet
import os
from seq2struct.commands.infer import Inferer
from seq2struct.datasets.spider import SpiderItem
from seq2struct.utils import registry
import torch
from seq2struct.datasets.spider_lib.preprocess.get_tables import dump_db_json_schema
from seq2struct.datasets.spider import load_tables_from_schema_dict
from seq2struct.utils.api_utils import refine_schema_names

import plotly.graph_objects as go
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
import sqlite3

os.environ["CUDA_VISIBLE_DEVICES"] = ""

exp_config = json.loads(
    _jsonnet.evaluate_file(
        "experiments/spider-configs/gap-run.jsonnet"))

model_config_path = exp_config["model_config"]
model_config_args = exp_config.get("model_config_args")

infer_config = json.loads(
    _jsonnet.evaluate_file(
        model_config_path,
        tla_codes={'args': json.dumps(model_config_args)}))

infer_config["model"]["encoder_preproc"]["db_path"] = "data/sqlite_files/"

inferer = Inferer(infer_config)

model_dir = exp_config["logdir"] + "/bs=12,lr=1.0e-04,bert_lr=1.0e-05,end_lr=0e0,att=1"
checkpoint_step = exp_config["eval_steps"][0]

model = inferer.load_model(model_dir, checkpoint_step)


class Data:
    db = 'aircraft'
    my_schema = dump_db_json_schema("data/sqlite_files/{db_id}/{db_id}.sqlite".format(db_id=db), db)
    schema, eval_foreign_key_maps = load_tables_from_schema_dict(my_schema)
    dataset = registry.construct('dataset_infer', {
        "name": "spider", "schemas": schema, "eval_foreign_key_maps": eval_foreign_key_maps,
        "db_path": "data/sqlite_files/"
    })
    spider_schema = dataset.schemas[db]


for _, schema in Data.dataset.schemas.items():
    model.preproc.enc_preproc._preprocess_schema(schema)


def infer(question):
    data_item = SpiderItem(
        text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
        code=None,
        schema=Data.spider_schema,
        orig_schema=Data.spider_schema.orig,
        orig={"question": question}
    )
    model.preproc.clear_items()
    enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)
    preproc_data = enc_input, None
    with torch.no_grad():
        output = inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)
    return output[0]["inferred_code"]


code = infer("List the vehicle flight number, date and pilot of all the flights, ordered by altitude.")
print(code)

bot = telebot.TeleBot('1708315176:AAFEz5KePXdG3MTomtMORAXjlE8YnBxwMsI')


def change_db():
    Data.my_schema = dump_db_json_schema("data/sqlite_files/{db_id}/{db_id}.sqlite".format(db_id=Data.db), Data.db)

    Data.schema, Data.eval_foreign_key_maps = load_tables_from_schema_dict(Data.my_schema)

    Data.dataset = registry.construct('dataset_infer', {
        "name": "spider", "schemas": Data.schema, "eval_foreign_key_maps": Data.eval_foreign_key_maps,
        "db_path": "data/sqlite_files/"
    })

    for _, schema in Data.dataset.schemas.items():
        model.preproc.enc_preproc._preprocess_schema(schema)

    Data.spider_schema = Data.dataset.schemas[Data.db]


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, 'Хеллоу, ' + message.from_user.first_name + '.')


@bot.message_handler(commands=['heroku', 'хероку'])
def send_welcome(message):
    bot.reply_to(message, 'Heroku не дремлет.')


@bot.message_handler(commands=['select'])
def url(message):
    button1 = KeyboardButton('aircraft')
    button2 = KeyboardButton('flight_1')
    button3 = KeyboardButton('flight_2')
    button4 = KeyboardButton('flight_4')
    button5 = KeyboardButton('pilot_record')

    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).row(
        button1, button2, button3, button4, button5)
    bot.send_message(message.chat.id, "Выбор БД.", reply_markup=markup)


@bot.message_handler(commands=['chart'])
def send_welcome(message):
    fig = go.Figure(
        data=[go.Bar(y=[2, 1, 3])],
    )
    bot.send_photo(message.chat.id, fig.to_image(format="png"), reply_to_message_id=message.id)


@bot.message_handler(commands=['check'])
def send_welcome(message):
    bot.reply_to(message, Data.db + '.')


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if 'aircraft' == message.text.lower():
        bot.reply_to(message, 'Ок, aircraft!\nХарактеристика самолетов и статистика пассажирооборота.')
        Data.db = 'aircraft'
        change_db()
    elif 'flight_1' == message.text.lower():
        bot.reply_to(message, 'Ок, flight_1!\n Информация о билетах.')
        Data.db = 'flight_1'
        change_db()
    elif 'flight_2' == message.text.lower():
        bot.reply_to(message, 'Ок, flight_2!\n Информация о компаниях и аэропортах.')
        Data.db = 'flight_2'
        change_db()
    elif 'flight_4' == message.text.lower():
        bot.reply_to(message, 'Ок, flight_4!\n Информация о перелетах и аэропортах.')
        Data.db = 'flight_4'
        change_db()
    elif 'pilot_record' == message.text.lower():
        bot.reply_to(message, 'Ок, pilot_record!\n Более точная информация о самолетах и пилотах.')
        Data.db = 'pilot_record'
        change_db()
    else:
        value = re.findall(r"'(.*?)'", message.text)
        answer = infer(message.text)
        if len(value) != 0:
            answer = answer.replace('terminal', value[0])
        bot.reply_to(message, answer)
        print(answer)
        try:
            conn = sqlite3.connect("data/sqlite_files/" + Data.db + '/' + Data.db + '.sqlite')
            cursor = conn.cursor()
            cursor.execute(answer)
            ans = cursor.fetchall()
            print(ans)
            final = ''
            for item in ans:
                if not (str(item[0]) in final):
                    final += str(item[0]) + '\n'
            bot.reply_to(message, final)
        except:
            print('Oh No!')


bot.polling(none_stop=True)
