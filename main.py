from PyQt5.uic import loadUi
from enum import unique
from operator import index
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QAction, QMainWindow, QSlider, QPushButton, QToolTip, QApplication, QTableWidgetItem

import os
import nltk # Library nltk
# nltk.download('stopwords')
import string # Library string
import re # Library regex
import pandas as pd # Library Pandax

#boolean
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import glob
import numpy as np
import sys
import math
#Stopwords = set(stopwords.words('english'))

#indexing data
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ======== Tokenize ========
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize

# ======== StopWord & Stemmer ========
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ========= Boolean ===========
from contextlib import redirect_stdout

class Ui_MainWindow(QMainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi('stkifix.ui', self)

        #Deklarasi
        self.savefile = []
        self.filename = []
        self.allWord = []
        self.token_word = []
        
        self.stopWord_word = []
        self.stemming_word = []
        self.counter = 1
        self.increase = 0

        #Function Button
        self.pushButton.clicked.connect(self.addFile)
        self.pushButton_2.clicked.connect(self.resetFile)
        self.pushButton_7.clicked.connect(self.preprocessingFile)
        self.pushButton_4.clicked.connect(self.tf_idf)
        self.pushButton_5.clicked.connect(self.jaccardFunction)
        self.pushButton_6.clicked.connect(self.ngramFunction)
        
    
    def addFile(self):
        self.fileName, fileType = QFileDialog.getOpenFileName(self.centralwidget, "Open File", "",
                                                              "*.txt;;All Files(*)")
        self.filePathName = os.path.basename(self.fileName)
        self.split = os.path.splitext(self.filePathName)[0]
        self.savefile.append(self.fileName)
        self.filename.append(self.split)
        self.listWidget.addItem(self.fileName)

        self.casefoldingWord(self.fileName)

        self.processingFile()
        self.token_word.extend(self.jadi)
        self.stopWord_word.extend(self.tokeStop)

        self.stemming_word.extend(self.tokenStem)
        self.stemming_word = list(dict.fromkeys(self.stemming_word))

        self.invertedIndex()
        self.printIncidence()

        self.counter += 1
        self.increase += 1

    def resetFile(self):
        self.counter = 1
        self.increase = 0
        self.stopWord_word.clear()
        self.stemming_word.clear()
        self.items_clear()

        self.tableWidget.clear()
        self.listWidget.clear()
        self.listWidget_2.clear()
        self.listWidget_3.clear()
        self.listWidget_4.clear()
        self.savefile.clear()

    def getFileNameOnly(self,filenameonly):
        filePathName = os.path.basename(filenameonly)
        splitName = os.path.splitext(filePathName)[0]
        return splitName

    def casefoldingWord(self,fileName):
       
        self.kalimatOpen = open(fileName,'r').read()
        self.tokenize = self.kalimatOpen.lower()
        # Menghapus angka
        self.angka = re.sub(r"\d+", "", self.tokenize)
        # Menghapus tanda baca
        self.tandabaca = self.angka.translate(str.maketrans("","",string.punctuation))
        # Menghapus whitespace
        self.whitespace = self.tandabaca.strip()
        # Menghapus beberapa whitespace menjadi whitespace tunggal
        self.whitespacetunggal = re.sub('\s+',' ',self.whitespace)

    def processingFile(self):
        # Case Folding -> Tokenizing -> Filtering (Stopword) -> Stemming
        # Tokenizing
        self.jadi = nltk.tokenize.word_tokenize(self.whitespacetunggal)
        self.frekuensi_tokens = nltk.FreqDist(self.jadi).most_common()


        #Filtering (StopWord)
        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        self.stop = stopword.remove(self.whitespacetunggal)   
        self.tokeStop = nltk.tokenize.word_tokenize(self.stop)

        # Stemming
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        self.stemming   = self.stemmer.stem(self.stop)
        self.tokenStem = nltk.tokenize.word_tokenize(self.stemming)
        # self.kalimatOpen = nltk.tokenize.word_tokenize(self.stop)    

    def preprocessingFile(self):
        self.listWidget_2.clear()

        self.fileNamePre = self.listWidget.currentItem().text()
        self.splitPre = self.getFileNameOnly(self.fileNamePre)
        self.listWidget_2.addItem(self.splitPre)
        self.casefoldingWord(self.fileNamePre)
        
        self.processingFile()
        
        #Add listWidget
        self.label_2.setText('Kalimat Text : {}'.format(self.kalimatOpen))
        self.listWidget_2.addItem('Token Kata : \n{}'.format(self.jadi))
        self.listWidget_2.addItem('\nFrekuensi Kata : \n{}'.format(self.frekuensi_tokens))
        self.listWidget_3.addItem('Stopword : \n{}'.format(self.stop))
        self.listWidget_4.addItem('Stemming : \n{}'.format(self.stemming))



        # self.counter += 1
        # self.increase += 1

    def invertedIndex(self):
        self.listWidget_7.clear()

        for item in range(len(self.stemming_word)):
            exist_file = list()
            index_data = list()
            count = 0
            for data in self.savefile:
                with open(data, 'r') as namaFile:
                    for isi in namaFile:
                        isi = isi.lower()  
                        tokenKata = nltk.tokenize.word_tokenize(isi)
                        if self.stemming_word[item] in isi:
                            exist_file.append(os.path.basename(data))
                            exist_file = list(dict.fromkeys(exist_file))
  
      
            
            self.listWidget_7.addItem('{}\t: <{}>'.format(self.stemming_word[item], exist_file))

    def printIncidence(self):

        self.items_clear()

        panjang_col = len(self.savefile)
        panjang_row = len(self.stemming_word)

        self.tableWidget_2.setColumnCount(panjang_col)
        self.tableWidget_2.setRowCount(panjang_row)

        self.tableWidget_2.setHorizontalHeaderLabels(self.filename)
        self.tableWidget_2.setVerticalHeaderLabels(self.stemming_word)

        for x in range(len(self.stemming_word)):
            for y in range(len(self.filename)):
                with open(self.savefile[y], 'r') as openFile:
                    for content in openFile:
                        if self.stemming_word[x] in content.lower():
                            self.tableWidget_2.setItem(x, y, QTableWidgetItem('1'))
                        else:
                            self.tableWidget_2.setItem(x, y, QTableWidgetItem('0'))         
    
    # ========= TF/IDF ===========

    def tf_idf(self):

        total = []
        for count in self.savefile:
            total.append(0)


        font = QFont()
        font.setBold(True)

        userInput = self.textEdit_2.toPlainText().lower()
        userInput = re.split(r'\W+', userInput)

        
        kolom_tf = ['df', 'D/df', 'IDF', 'IDF+1']

        jarak_W = len(self.savefile) + len(kolom_tf)
        panjang_kolom = len(self.savefile)*2 + len(kolom_tf) + 1

        self.tableWidget.setColumnCount(panjang_kolom)
        self.tableWidget.setRowCount(len(userInput)+3)
        
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setVisible(False)

        # ========== Span Tf ==========

        ''' Format bikin span : tableTf.setSpan(row, column, rowSpan, columnSpan) '''

        self.tableWidget.setSpan(0, 1, 1, len(self.savefile))
        newItem = QTableWidgetItem("tf")
        newItem.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setItem(0, 1, newItem)
        self.tableWidget.item(0, 1).setFont(font)

        # ========== Span df ==========

        self.tableWidget.setSpan(0, len(self.savefile)+1, 2, 1)
        newItem = QTableWidgetItem("df")
        newItem.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setItem(0, len(self.savefile)+1, newItem)
        self.tableWidget.item(0, len(self.savefile)+1).setFont(font)

        # ========== Span D/df ==========

        self.tableWidget.setSpan(0, len(self.savefile)+2, 2, 1)
        newItem = QTableWidgetItem("D / df")
        newItem.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setItem(0, len(self.savefile)+2, newItem)
        self.tableWidget.item(0, len(self.savefile)+2).setFont(font)

        # ========== Span IDF ==========

        self.tableWidget.setSpan(0, len(self.savefile)+3, 2, 1)
        newItem = QTableWidgetItem("IDF")
        newItem.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setItem(0, len(self.savefile)+3, newItem)
        self.tableWidget.item(0, len(self.savefile)+3).setFont(font)

        # ========== Span IDF+1 ==========

        self.tableWidget.setSpan(0, len(self.savefile)+4, 2, 1)
        newItem = QTableWidgetItem("IDF")
        newItem.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setItem(0, len(self.savefile)+4, newItem)
        self.tableWidget.item(0, len(self.savefile)+4).setFont(font)

        # ========== Span W ==========

        self.tableWidget.setSpan(0, len(self.savefile)+5, 1, len(self.savefile))
        newItem = QTableWidgetItem("W = tf*(IDF+1)")
        newItem.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setItem(0, len(self.savefile)+5, newItem)
        self.tableWidget.item(0, len(self.savefile)+5).setFont(font)

        # ========== MAKE TABLE ==========

        # __Print Document di kolom tf (kiri)__ 
        for y in range(len(self.savefile)):
            cell_item = QTableWidgetItem(str(self.filename[y]))
            cell_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tableWidget.setItem(1, y+1, cell_item)
            self.tableWidget.item(1, y+1).setFont(font)
            
        # __Print Document di kolom tf (kanan)__
        for y in range(len(self.savefile)):
            cell_item = QTableWidgetItem(str(self.filename[y]))
            cell_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tableWidget.setItem(1, y+1+jarak_W, cell_item)
            self.tableWidget.item(1, y+1+jarak_W).setFont(font)
        
        # # __Isi Tf-Idf__
        for x in range(len(userInput)): # 0 < x < 3 (x = 0)

            exist_in = []

            for y in range((panjang_kolom)): # 0 < y < 11 (y = 0)

                # __Print userInput di row__
                self.tableWidget.setItem(x+2, 0, QTableWidgetItem(str(userInput[x])))
                
                # __Print nilai tf per document__
                if y < len(self.savefile):
                    with open(self.savefile[y], 'r') as openFile:
                        for content in openFile:
                            if userInput[x] in content.lower():
                                exist_in.append(1)
                            else:
                                exist_in.append(0)

                    self.tableWidget.setItem(x+2, y+1, QTableWidgetItem(str(exist_in[y])))

                # __df__
                df = 0
                for count in exist_in:
                    if count > 0:
                        df = df + 1 # df = 1

                # __D/df__
                if df != 0:
                    D = round(len(self.savefile) / df, 3)
                else:
                    D = 1

                # __idf__
                idf = round(math.log(D), 3)

                # __idf+1__
                idf_1 = round(idf+1,3)

                # __W__
                W = []
                for freq in range(len(exist_in)):
                    W.append(round(exist_in[freq]*(idf+1), 3))

                if len(W) == len(self.savefile) and y == len(self.savefile):
                    # __SUM__
                    zipped_list = zip(total, W)
                    total = [x+y for (x, y) in zipped_list]
                                    
                # __Print nilai setelah tf dan sebelum W__
                if y == (len(self.savefile)+1) and y <= jarak_W:
                    self.tableWidget.setItem(x+2, y, QTableWidgetItem(str(df)))
                    self.tableWidget.setItem(x+2, y+1, QTableWidgetItem(str(D)))
                    self.tableWidget.setItem(x+2, y+2, QTableWidgetItem(str(idf)))
                    self.tableWidget.setItem(x+2, y+3, QTableWidgetItem(str(idf_1)))
                
                # __Print nilai W per dokumen__
                if y > jarak_W:
                    self.tableWidget.setItem(x+2, y, QTableWidgetItem(str(W[y-(jarak_W+1)])))
                
                if x == len(userInput)-1: # x = 2. userInput = 2

                    if y == jarak_W: # y = 7
                        item_sum = QTableWidgetItem('Sum :')
                        self.tableWidget.setItem(x+3, y, item_sum)
                        self.tableWidget.item(x+3, y).setFont(font)
                    elif y > jarak_W:
                        total_sum = QTableWidgetItem(str(total[y-(jarak_W+1)]))
                        self.tableWidget.setItem(x+3, y, total_sum)
                        self.tableWidget.item(x+3, y).setFont(font)

        file_rank = self.filename.copy()

        self.listWidget_8.addItem('Ranking:')

        for i in range(len(self.savefile)-1):
            for j in range(len(self.savefile)-i-1):
                if total[j] < total[j+1]:
                    temp_total = total[j]
                    total[j] = total[j+1]
                    total[j+1] = temp_total

                    temp = file_rank[j]
                    file_rank[j] = file_rank[j+1]
                    file_rank[j+1] = temp

        print(type(file_rank))
        for i in range(len(file_rank)):
            print(i)
            self.listWidget_8.addItem('{}. {}'.format(i,file_rank[i]))



        # print(total)
        # print(file_rank)

    def jaccardFunction(self):
        self.listWidget_6.clear()
        self.listWidget_11.clear()
        total = []
        for count in self.savefile:
            total.append(count)

    
        userInput = self.textEdit_3.toPlainText().lower()
        # print(userInput)
        userInput = re.split(r'\W+', userInput)


        a = userInput
        self.listWidget_6.addItem('Orignal Text : {}'.format(a))
        increaseNo = 0
        hasil_urutJaccard = list()
        for folder in total:
            self.listWidget_6.addItem('==========================================')
            with open(folder,'r') as namafile:
                for isi in namafile:
                    isi = isi.lower()
                    isi = isi.split()
                    self.listWidget_6.addItem('{} :'.format(self.filename[increaseNo]))
                    self.listWidget_6.addItem('Original Text : [{}]'.format(isi))
                    c = set(a).intersection(isi)
                    d = round(float(len(c)) / (len(a)+len(isi)-len(c)),3)
                    self.listWidget_6.addItem('{} / {} : [{}]'.format(len(c),(len(a)+len(isi)-len(c)),d))
                    hasil_urutJaccard.append([self.filename[increaseNo],d])
                    increaseNo +=1
           
        # print(d)

        hasil_urutJaccard.sort(key=lambda row: (row[1]),reverse=True)

        increaseNo = 0
        self.listWidget_11.addItem('Ranking')
        for folder in total:
            self.listWidget_11.addItem('{} : [{}]'.format(hasil_urutJaccard[increaseNo][0],hasil_urutJaccard[increaseNo][1]))
            increaseNo+=1

        # numpyHasil = np.array(hasil_urut).sort()
        # print(numpyHasil)
    
        # for i in range(len(numpy)-1):
        #     for j in range(len(self.savefile)-i-1):
        #         if total[j] < total[j+1]:
        #             temp_total = total[j]
        #             total[j] = total[j+1]
        #             total[j+1] = temp_total

        #             temp = file_rank[j]
        #             file_rank[j] = file_rank[j+1]
        #             file_rank[j+1] = temp


    def ngramFunction(self):
        self.listWidget_9.clear()
        self.listWidget_10.clear()
        total = []
        for count in self.savefile:
            total.append(count)

    
        userInput = self.textEdit_4.toPlainText().lower()
        # print(userInput)
        userInput = re.split(r'\W+', userInput)

        m = 2 # nilai untuk jumlah grams
        nGramSearchList = self.getNGrams(userInput,m)
        
        print(nGramSearchList)
        self.listWidget_9.addItem('Orignal Text : {}'.format(nGramSearchList))



        increaseNo = 0
        hasil_urutnGram = list()
        for folder in total:
            self.listWidget_9.addItem('==========================================')
            for i in range(len(nGramSearchList)):
                with open(folder,'r') as namafile:
                    a = 0
                    for isi in namafile:
                        isi = isi.lower()
                        isi = isi.split()
                        fileGetNGrams = self.getNGrams(isi,m)
                        self.listWidget_9.addItem('{} : \n{}'.format(self.filename[increaseNo],fileGetNGrams))
                        
                        # print(len(fileGetNGrams)) // Jumlah panjang listnya
                        for j in range(len(fileGetNGrams)):
                            if nGramSearchList[i] == fileGetNGrams[j]:
                                a +=1
            d = round(float(a)/(len(isi)),3)
            hasil_urutnGram.append([self.filename[increaseNo],d])
            self.listWidget_9.addItem('{} / {}: {}'.format(a,len(isi),d))
            increaseNo +=1
            

        hasil_urutnGram.sort(key=lambda row: (row[1]),reverse=True)

        increaseNo = 0
        self.listWidget_10.addItem('Ranking')
        for folder in total:
            self.listWidget_10.addItem('{} : [{}]'.format(hasil_urutnGram[increaseNo][0],hasil_urutnGram[increaseNo][1]))
            increaseNo+=1

        


    def getNGrams(self,wordlist,n):
        ngrams = []
        for i in range(len(wordlist)-(n-1)):
            ngrams.append(wordlist[i:i+n])
        return ngrams

    def items_clear(self):
        for item in self.tableWidget_2.selectedItems():
            newitem = QTableWidgetItem()
            self.tableWidget_2.setItem(item.row(), item.column(), newitem)

    


app = QApplication([])
window = Ui_MainWindow()
window.show()
app.exec_()