#!/usr/bin/python3
#--coding:UTF-8--

import random
import requests
import re

class ProdictTwoColorBalls:
    RED_NUMS  = 6
    BLUE_NUMS = 1

    def __init__(self):
        self.red_bucket = [ x for x in range(1, 34)]
        self.blue_bucket = [x for x in range(1, 17)]
        self.result = []
        self.history_result = []

    def predict_dst(self, dst_two_color_balls, predict):
        nums = 0
        while True:
            #two_color_balls = self.predict_random()
            two_color_balls = predict()
            nums += 1
            print("[",nums,"]:", two_color_balls)
            if two_color_balls == dst_two_color_balls:
                break
        return nums

    def predict_random(self):
        red_bucket = self.red_bucket
        blue_bucket = self.blue_bucket
        result = []
        red_balls = random.sample(red_bucket, self.RED_NUMS)
        red_balls.sort()
        for blue in blue_bucket:
            blue_ball = random.choice(blue_bucket)
            if blue_ball not in red_balls:
                break
        for ball in red_balls:
            result.append(ball)
        result.append(blue_ball)
        return result

    def predict(self, nums):
        red_bucket = self.red_bucket[:]
        blue_bucket = self.blue_bucket[:]

        for index in range(nums):
            red_balls = random.sample(red_bucket, self.RED_NUMS)
            red_balls.sort()
            for blue in blue_bucket:
                blue_ball = random.choice(blue_bucket)
                if blue_ball not in red_balls:
                    break
            for val in red_balls:
               red_bucket.remove(val)
            blue_bucket.remove(blue_ball)
            self.result.append([red_balls, blue_ball])
            print(self.result[0]);
        # print("remaine red_bucket:", red_bucket)
        # print("remaine blue_bucket:", blue_bucket)

    def print_result(self):
        print('---------------------------------------------')
        for i, result in enumerate(self.result):
            print("[",i,"] ", result)
        print('---------------------------------------------')
    
    def get_history_from_url(self, url):
        r = requests.get(url)
        r.encoding = 'gbk'
        res = r.text
        no = "".join(re.findall(r'<font class="cfont2"><strong>(.+?)</strong>',res))
        red = ",".join(re.findall(r'<li class="ball_red">(.+?)</li>',res))
        blue = "".join(re.findall(r'<li class="ball_blue">(.+?)</li>',res))
        return no, red, blue

    def save_history_result(self, filename, begin, end):
        with open(filename, 'w') as fout:
            for i in range(begin, end):
                url = "http://kaijiang.500.com/shtml/ssq/"+str(i)+".shtml"
                no, red, blue = self.get_history_from_url(url)
                if (len(red) == 0):
                    continue
                res = "第" + no + "期：" + red + " "+ blue
                print(res)
                fout.write(res + '\n')

    def load_history_from_file(self, filename):
        with open(filename, 'r') as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                elems = line.split("：")
                # print(elems[0])
                # print(elems[1])
                red_balls_str, blue_balls = elems[1].split(" ")
                red_balls = red_balls_str.split(",")
                # print(red_balls)
                # print(blue_balls)
                two_color_balls = red_balls
                two_color_balls.append(blue_balls.split('\n')[0])
                # print(line, " -> " , two_color_balls)
                self.history_result.append(two_color_balls)

    def print_history(self):
        for i, res in enumerate(self.history_result):
            print("%d, %s" %(i, res))

    def analyze_history(self):
        print("add analyze_history ...")

if __name__ == "__main__":
    print("predict two color balls")
    prodictTcolor = ProdictTwoColorBalls()
    # prodictTcolor.predict(1)
    # prodictTcolor.print_result()
    # prodictTcolor.getHistRes("http://kaijiang.500.com/shtml/ssq/22025.shtml")
    # prodictTcolor.saveHistRes("history.txt", 20000, 22026)

    prodictTcolor.load_history_from_file("history.txt")
    prodictTcolor.print_history()
    prodictTcolor.analyze_history()

    # dst_two_color_balls = [14, 15, 18, 19, 26, 32, 9]
    # nums = prodictTcolor.predict_dst(dst_two_color_balls, prodictTcolor.predict_random)
    # print("预测号码：", dst_two_color_balls, " 需要 ", nums, " 次数")