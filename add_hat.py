#! /usr/bin/env python
# coding: utf-8
from __future__ import print_function
# 
import os, time
import numpy as np 
import cv2
import dlib

from functools import partial
from queue import Queue
# import pytest f
import logging
from wxpy import *

import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='.log',
                filemode='w')

_base_dir = os.path.dirname(os.path.realpath(__file__))
attachments_dir = os.path.join(_base_dir, 'attachments')
avtar_dir = os.path.join(attachments_dir, 'avtar')
xmas_dir = os.path.join(attachments_dir, 'xms')
gen_attachment_path = partial(os.path.join, attachments_dir)
error_msg = u'未检测出人脸，请上传正面照：）'

# 初始化机器人，扫码登陆
_bot = Bot(False, True)
_friend = ensure_one(_bot.friends().search(u'肖长省'))
# _group = ensure_one(_bot.groups().search(u'trade-test'))
# _member = ensure_one(_group.search(u'trade-ripple'))
# _shared_dict = dict()

# global_use = partial(pytest.fixture, scope='session', autouse=True)


# 给img中的人头像加上圣诞帽，人脸最好为正脸
def add_hat(img,hat_img):
    # 分离rgba通道，合成rgb三通道帽子图，a通道后面做mask用
    r,g,b,a = cv2.split(hat_img) 
    rgb_hat = cv2.merge((r,g,b))

    cv2.imwrite("hat_alpha.jpg",a)

    # ------------------------- 用dlib的人脸检测代替OpenCV的人脸检测-----------------------
    # # 灰度变换
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    # # 用opencv自带的人脸检测器检测人脸
    # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")                       
    # faces = face_cascade.detectMultiScale(gray,1.05,3,cv2.CASCADE_SCALE_IMAGE,(50,50))

    # ------------------------- 用dlib的人脸检测代替OpenCV的人脸检测-----------------------

    # dlib人脸关键点检测器
    predictor_path = "shape_predictor_5_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)  

    # dlib正脸检测器
    detector = dlib.get_frontal_face_detector()

    # 正脸检测
    dets = detector(img, 1)

    # 如果检测到人脸
    if len(dets)>0:  
        for d in dets:
            x,y,w,h = d.left(),d.top(), d.right()-d.left(), d.bottom()-d.top()
            # x,y,w,h = faceRect  
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)

            # 关键点检测，5个关键点
            shape = predictor(img, d)
            # for point in shape.parts():
            #     cv2.circle(img,(point.x,point.y),3,color=(0,255,0))

            # cv2.imshow("image",img)
            # cv2.waitKey()  

            # 选取左右眼眼角的点
            point1 = shape.part(0)
            point2 = shape.part(2)

            # 求两点中心
            eyes_center = ((point1.x+point2.x)//2,(point1.y+point2.y)//2)

            # cv2.circle(img,eyes_center,3,color=(0,255,0))  
            # cv2.imshow("image",img)
            # cv2.waitKey()

            #  根据人脸大小调整帽子大小
            factor = 1.5
            resized_hat_h = int(round(rgb_hat.shape[0]*w/rgb_hat.shape[1]*factor))
            resized_hat_w = int(round(rgb_hat.shape[1]*w/rgb_hat.shape[1]*factor))

            if resized_hat_h > y:
                resized_hat_h = y-1

            # 根据人脸大小调整帽子大小
            resized_hat = cv2.resize(rgb_hat,(resized_hat_w,resized_hat_h))

            # 用alpha通道作为mask
            mask = cv2.resize(a,(resized_hat_w,resized_hat_h))
            mask_inv =  cv2.bitwise_not(mask)

            # 帽子相对与人脸框上线的偏移量
            dh = 0
            dw = 0
            # 原图ROI
            # bg_roi = img[y+dh-resized_hat_h:y+dh, x+dw:x+dw+resized_hat_w]
            bg_roi = img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)]

            # 原图ROI中提取放帽子的区域
            bg_roi = bg_roi.astype(float)
            mask_inv = cv2.merge((mask_inv,mask_inv,mask_inv))
            alpha = mask_inv.astype(float)/255

            # 相乘之前保证两者大小一致（可能会由于四舍五入原因不一致）
            alpha = cv2.resize(alpha,(bg_roi.shape[1],bg_roi.shape[0]))
            # print("alpha size: ",alpha.shape)
            # print("bg_roi size: ",bg_roi.shape)
            bg = cv2.multiply(alpha, bg_roi)
            bg = bg.astype('uint8')

            cv2.imwrite("bg.jpg",bg)
            # cv2.imshow("image",img)
            # cv2.waitKey()

            # 提取帽子区域
            hat = cv2.bitwise_and(resized_hat,resized_hat,mask = mask)
            cv2.imwrite("hat.jpg",hat)
            
            # cv2.imshow("hat",hat)  
            # cv2.imshow("bg",bg)

            # print("bg size: ",bg.shape)
            # print("hat size: ",hat.shape)

            # 相加之前保证两者大小一致（可能会由于四舍五入原因不一致）
            hat = cv2.resize(hat,(bg_roi.shape[1],bg_roi.shape[0]))
            # 两个ROI区域相加
            add_hat = cv2.add(bg,hat)
            # cv2.imshow("add_hat",add_hat) 

            # 把添加好帽子的区域放回原图
            img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)] = add_hat

            # 展示效果
            # cv2.imshow("img",img )  
            # cv2.waitKey(0)  

            return img

def add_hat_file(in_img, hat_img='hat2.png'):
    # 读取帽子图，第二个参数-1表示读取为rgba通道，否则为rgb通道
    hat_img = cv2.imread(hat_img, -1)

    # 读取头像图
    img = cv2.imread(in_img)
    output = add_hat(img,hat_img)

    # 展示效果
    # cv2.imshow("output", output)  
    # cv2.waitKey(0)  
    out_img = os.path.join(xmas_dir, os.path.basename(in_img))
    cv2.imwrite(out_img, output)
    # import glob as gb 

    # img_path = gb.glob("./images/*.jpg")

    # for path in img_path:
    #     img = cv2.imread(path)

    #     # 添加帽子
    #     output = add_hat(img,hat_img)

    #     # 展示效果
    #     cv2.imshow("output",output )  
    #     cv2.waitKey(0)  

    cv2.destroyAllWindows()
    logging.info(out_img)
    return out_img


# 自动接受新的好友请求
@_bot.register(msg_types=FRIENDS)
def auto_accept_friends(msg):
    # 接受好友请求
    new_friend = msg.card.accept() 
    try:
        avtar_path = os.path.join(avtar_dir, new_friend.uin() + '.jpg')
        avatar = new_friend.get_avatar(avtar_path)
        logging.debug(avtar_path)
        logging.debug(avatar)
        xmas_img = add_hat_file(avtar_path)
        logging.debug(xmas_img)
        new_friend.send(u'欢迎朋友，发送“圣诞”、“xms”、“christmas”或者靓照自动送帽子.全能机器人陪聊')
        new_friend.send_image(xmas_img)
    except Exception as e:
        logging.exception(e)
        new_friend.send(error_msg)
        # raise e
    

# 自动回复图片
@_bot.register(msg_types=PICTURE)
def auto_reply_picture(msg):
    # 向好友发送消息
    try:
        avtar_path = os.path.join(avtar_dir, str(msg.id) + '.jpg')
        avatar = msg.get_file(avtar_path)
        logging.debug(avtar_path)
        logging.debug(avatar)
        xmas_img = add_hat_file(avtar_path)
        logging.debug(xmas_img)
        # msg.reply(u'正为你戴上圣诞帽...')
        msg.reply_image(xmas_img)
    except Exception as e:
        logging.exception(e)
        msg.reply(error_msg)
        # raise e
        
# 关键字处理
@_bot.register(msg_types=TEXT)
def auto_reply_keywords(msg):
    if msg.text.find(u'圣诞') > -1 or msg.text.find(u'xms') > -1 or msg.text.find(u'christmas') > -1:
        # 向好友发送消息
        try:
            avtar_path = os.path.join(avtar_dir, str(msg.id) + '.jpg')
            avatar = msg.chat.get_avatar(avtar_path)
            logging.debug(avtar_path)
            logging.debug(avatar)
            xmas_img = add_hat_file(avtar_path)
            logging.debug(xmas_img)
            # msg.reply(u'正为你戴上圣诞帽...')
            msg.reply_image(xmas_img)
        except Exception as e:
            logging.exception(e)
            msg.reply(error_msg)
            # raise e       
    else:
        tuling = Tuling(api_key='42bbff0b64664a1a8014466d7c374352')
        tuling.do_reply(msg)

# 进入 Python 命令行、让程序保持运行
embed(local=None, banner=u'进入命令行', shell='python')

# 或者仅仅堵塞线程
# bot.join()

