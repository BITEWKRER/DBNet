# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import datetime
import logging

import pytz

from configs import config

if config.train:
    log_name_path = f'{config.basePath}/{config.mode}_{config.dataset}_train.txt'
else:
    log_name_path = f'{config.basePath}/{config.mode}_{config.dataset}_{config.log_name}.txt'


logger = logging.getLogger('info')
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(log_name_path)
console = logging.StreamHandler()

handler.setLevel(logging.INFO)
console.setLevel(logging.INFO)

formatter = logging.Formatter('%(time)s - %(message)s')
handler.setFormatter(formatter)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)


def logs(info):
    # 获取UTC时间
    utc_now = datetime.datetime.utcnow()
    # 转换为您所在的时区
    local_tz = pytz.timezone('Asia/Shanghai')
    now = utc_now.replace(tzinfo=pytz.utc).astimezone(local_tz)

    logger.info(info, extra={'time': now.strftime('%Y-%m-%d %H:%M:%S')})

# from loguru import logger
#
# logger.remove()
# logger.add(log_name_path, format="{message}", rotation="500 MB", enqueue=True, level="INFO")
#
#
# def logs(info):
#     utc_now = datetime.datetime.utcnow()
#     # 转换为您所在的时区
#     local_tz = pytz.timezone('Asia/Shanghai')
#     now = utc_now.replace(tzinfo=pytz.utc).astimezone(local_tz)
#     logger.info(f"{now.strftime('%Y-%m-%d %H:%M:%S')} - " + info)
