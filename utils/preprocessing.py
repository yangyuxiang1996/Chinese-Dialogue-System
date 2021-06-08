#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-05-20 16:12:22
LastEditors: yangyuxiang
LastEditTime: 2021-06-05 23:04:53
FilePath: /Chinese-Dialogue-System/utils/preprocessing.py
'''

import sys
sys.path.append('..')
import os
import re
import pandas as pd
from config import Config
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


def filter_content(sentence):
    """
    特殊字段有：
    1. #E-s[数字x] #E-2[数字x] 等一系列数字—— 表情
    2. [ORDERID_10187709] —— 订单号
    3. [数字x] —— 数字
    4. https://item.jd.com/5898522.html —— 网址
    5. [地址x] —— 地址
    6. [链接x] —— 链接
    7. [金额x] —— 金额
    8. [日期x] —— 日期
    9. [时间x] —— 时间
    10. [站点x] —— 站点
    11. [组织机构x] ——组织机构
    12. [电话x] —— 电话
    13. [姓名x] —— 人名
    对于表情，做法是直接删除。其他用希腊符号替换。
    """
    sep = Config.sep
    if isinstance(sentence, str):
        sentence = [sentence]
    sentence = sep.join(sentence)
    sentence = re.sub(
        r"#E\-[\w]*(抱拳|傲慢|得意|蛋糕|呕吐|闭嘴|礼物|yaoping|柠檬|流泪|怒火|撇嘴|太阳|咒骂|糗|猪猪|足球|磕头|大兵|电话|灯泡|飞鸟|奋斗|高兴|击打|饥饿|咖啡|口罩|骷髅|可乐|疯狂|白眼|阴险|叹气|奸笑|发呆|害羞|飞吻|怒火|悲伤|胜利|生病|弱|可怜|咖啡|酷酷|眩晕|流泪|发抖|难过|右哼哼|惊恐|悲伤|犯困|愤怒|凋谢|哈欠|拥抱|抓狂|鄙视|时间|啤酒|勾引|左哼哼|月亮|偷笑|震惊|惊讶|跳跳|瞌睡|可爱|衰样|好|憨笑|水果|色色|黑线|微笑|流汗|握手|心碎|问号|大哭|亲亲|抠鼻|拜拜|鬼脸|香吻|米饭|花朵|尴尬|擦汗|安慰|委屈|调皮|爱心|我一定尽力为您解答的哦|很棒|鼓掌)+",
        "α", sentence) 
    sentence = re.sub(r"#E\-[\w]+\[数字x]", "α", sentence)
    sentence = re.sub(r"\[ORDERID_[\d]+]", "[订单x]", sentence)
    sentence = re.sub(r"\[数字x]", "γ", sentence)
    sentence = re.sub(r"\[链接x]", "ε", sentence)
    sentence = re.sub(r"\[表情]", "α", sentence)
    sentence = re.sub("<sep>", sep, sentence)
    sentence = re.sub("<SEP>", sep, sentence)
    sentence = re.sub(
        r"(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?",
        "ε", sentence)
    sentence = re.sub(r"(http|ftp|https):\/\/ε", "ε", sentence)
    sentence = re.sub(r"[\d]+.*[\d]+", "γ", sentence)
    sentence = re.sub(r"【收到不支持的消息类型，暂无法显示】", " ", sentence)

    sentence = re.sub(r"#E\-[s]*(ν|γ|π|ζ|ρ|α|ε)*", "α", sentence)
    sentence = re.sub("α", " ", sentence)
    sentence = re.sub("ε", "[链接x]", sentence)
    sentence = re.sub("γ", "[数字x]", sentence)

    return sentence


def read_file(path, is_train=False):
    '''
    @description: 读取文件， 并将原始数据中同一个人多次输入合并为一句，用sep分隔符隔开。
    @param {type}
    path: 数据文件所在目录
    is_train： 是否为训练数据集
    @return:list  包含session_id, role, content
    '''
    chat = []

    with open(path, 'r') as f:

        tmp = []
        sessions = set()
        session_id, custom_id, is_assistance, content = '', '', '', []
        for lines in f:
            if len(sessions) > 50000:  # 50k sessions at most.
                break
            line = lines.strip().replace(' ', '').split('\t')
            if len(line) < 5:  # Filtering short samples.
                continue
            if is_train:
                session_id_next, custom_id_next, is_assistance_next = \
                    line[0], line[1], line[2]
            else:
                session_id_next, custom_id_next, is_assistance_next = \
                    line[2], line[1], line[3]
            sessions.add(session_id_next)
            if session_id != session_id_next and session_id != '':  # 当前会话id结束, 把上一个会话的最后记录存起来
                fc = filter_content(content)
                if fc != '':
                    tmp.append([
                        session_id, 'custom'
                        if str(is_assistance) == '0' else 'assistance', fc
                    ])
                    content = []
                chat.extend(tmp)
                tmp = []
                session_id, custom_id = session_id_next, custom_id_next
            else:
                if is_assistance != is_assistance_next and \
                        is_assistance != '': # 连着两句话不是一个人说的
                    content = filter_content(content)
                    is_assistance = 'custom' if str(
                        is_assistance) == '0' else 'assistance'
                    if content != '':
                        tmp.append([session_id, is_assistance, content])
                    is_assistance = is_assistance_next
                    content = [line[-1]]
                else:
                    content.append(line[-1]) # 连着两句话是同一个人说的  
                    is_assistance = is_assistance_next
                    session_id, _ = session_id_next, custom_id_next
    if content != '':
        tmp.append([
            session_id,
            'custom' if str(is_assistance) == '0' else 'assistance',
            filter_content(content)
        ])
    chat.extend(tmp)
    return chat


def clean(sent, sep='<'):
    '''
    @description: 过滤无用符号， 并对[SEP] 等分割符号， 假如前后空格，避免影响分词结果
    @param {type}
    sent: 句子
    sep: 分隔符是以< or [ 开头
    @return: string 清洗后的句子
    '''
    sent = re.sub(r"[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+",
                  "", sent)
    i = 0
    tmp = []
    while i < len(sent):
        if sent[i] != sep:
            tmp.append(sent[i])
            i += 1
        else:
            tmp.append(sent[i:i + 5])
            i += 5
    # 过滤短文本？
    return " ".join(tmp)


def generate_data(filepath, save=True, to_file=None, pair=False):
    '''
    @description: 将read_file 的结果进行转化， 问答pair, 或者将一次会话内容合并一起
    例如：
    before：
    0      0    2297112      custom                                       我收到商品不知道怎么使用
    1      0    2297112  assistance                   您好，京东客服[数字x]**号很高兴为您服务！[SEP]NULL
    2      2    2297112      custom                                        我买的数据线充不进去电
    3      2    2297112  assistance  [数字x]PlusiPadAir/Pro!@@@!外观有破损吗[SEP]您好，您收到商品是否...

    after：
    0      0    2297112                   您好，京东客服[数字x]**号很高兴为您服务！[SEP]NULL         我收到商品不知道怎么使用
    1      2    2297112  [数字x]PlusiPadAir/Pro!@@@!外观有破损吗[SEP]您好，您收到商品是否...          我买的数据线充不进去电

    @param {type}
    file_path， 原始数据路径
    save, 是否保存
    to_file， 保存的文件名， 会根据文件名判断是否为训练集
    pair: 是否生成问答pair的结果
    @return: 处理后可作为模型输入的csv数据集。
    '''

    data = read_file(filepath, 'train' in to_file)
    data = pd.DataFrame(data, columns=['session_id', 'role', 'content'])
    if 'train' in to_file:
        data = data[(data['content'].str.len() <= 128)
                    & (data['content'].str.len() > 1)].reset_index(drop=True)

    data = data.reset_index()
    data['index'] = data['index'].apply(lambda x: x - 1
                                        if x % 2 == 1 else x)
    logging.info(data.head(5))

    data = data.pivot_table(index=['index', 'session_id'],
                            columns='role',
                            values='content',
                            aggfunc='first').reset_index()
    logging.info(data.head(5))
    data = data[['session_id', 'custom',
                'assistance']].dropna().reset_index(drop=True)

    logging.info(data.head(5))

    if save:
        data.to_csv('{}.csv'.format(to_file), index=False)
    return data


if __name__ == "__main__":
    train_raw = Config.train_raw
    dev_raw = Config.dev_raw
    test_raw = Config.test_raw
    root_path = Config.root_path

    dev = generate_data(dev_raw,
                        save=True,
                        to_file=os.path.join(root_path, 'data/dev'),
                        pair=True)
    logging.info('Dev set created.')
    test = generate_data(test_raw,
                         save=True,
                         to_file=os.path.join(root_path, 'data/test'),
                         pair=True)
    logging.info('test set created.')
    data = generate_data(train_raw,
                         save=True,
                         to_file=os.path.join(
                             root_path, 'data/train_no_blank'),
                         pair=True)
    logging.info('training set created.')
