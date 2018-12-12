import jieba
import pymysql


# 数据库取数据
def getdata(website_num):
    db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='scrapy',
                         charset="utf8")
    cursor = db.cursor()
    cursor.execute('select city,salary,jd,job_url from details_3000 limit %s', (website_num,))
    rows = cursor.fetchall()

    db.commit()
    db.close()

    tup = []
    for x in rows:
        tup.append((x[2], x[-1], x[0]+x[1]))
    print('spider返回样本数量：', len(tup))

    return tup  # [(),(),...]


# 分词
def cut_words(rows):
    words, the_type_doc = [], []
    for html in rows:
        url, typ = html[1], html[2]
        # 去HTML
        # html = re.sub("[^\u4e00-\u9fa5]+", '', html[0])
        # 分词
        segs = jieba.cut(html[0], cut_all=False)
        html_word = list(segs)
        # 去停用词
        html_word = remove_stopwords(html_word)

        # 可用词汇少于2个，跳过这个样本
        if len(html_word) < 2:
            print("【cut_words】可用词汇少于2个，跳过这个样本", html_word)
            continue
        the_type_doc.append((url, typ, html_word))
        words.extend(html_word)
    print('分词、过滤停用词后的长度为：', len(words))

    # 总词列表 | 提特征值时所用到的样本
    return words, the_type_doc


# 去停用词
def remove_stopwords(words):
    ''' 去除停用词，返回新词频列表 '''
    # 加载停用词列表
    stopwords = []
    for word in open('stop.txt', 'r'):
        stopwords.append(word.strip())

    # 去除停用词后的列表
    new_word = []
    for word in words:
        if word not in stopwords and len(word) > 1:
            new_word.append(word)
    return new_word
