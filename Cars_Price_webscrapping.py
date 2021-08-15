#!/usr/bin/env python
# coding: utf-8

# In[1]:


import selenium
from selenium import webdriver
import time
from selenium.common.exceptions import NoSuchElementException,TimeoutException


# In[2]:


#First we will connect to webdriver
driver=webdriver.Chrome(r'C:\Users\Nitin Singh Tatrari\Downloads\chromedriver_win32\chromedriver.exe')


# In[3]:


#Open the webpage with webdriver
driver.get('https://www.olx.in/')


# In[4]:


#Clicking 
click1=driver.find_element_by_xpath("//a[@class='_2fitb']").click()


# In[5]:


#Clicking at 'Maharashtra'
click2=driver.find_element_by_xpath("//ul[@class='_3eZBP']/li[1]").click()


# In[7]:


#Laoding data 
for i in range(0,50):
    try:
        click1=driver.find_element_by_xpath("//button[@class='rui-39-wj rui-3evoE rui-1JPTg']").click()
        time.sleep(5)
    except  NoSuchElementException:
        print("Cannot load more")
        break


# In[8]:


#Storing URLs
URL=[]


# In[9]:


#Extracting the URls
urls=driver.find_elements_by_xpath("//li[@class='EIR5N']/a")
for url in urls:
    URL.append(url.get_attribute('href'))


# In[10]:


print(len(URL))


# In[11]:


#Clicking back to "India"
click3=driver.find_element_by_xpath("//a[@class='_396bV _2zs4P']").click()


# In[12]:


#Clicking 'Tamil Nadu'
click4=driver.find_element_by_xpath("//ul[@class='_3eZBP']/li[2]").click()


# In[14]:


#Laoding data 
for i in range(0,50):
    try:
        click1=driver.find_element_by_xpath("//button[@class='rui-39-wj rui-3evoE rui-1JPTg']").click()
        time.sleep(5)
    except  NoSuchElementException:
        print("Cannot load more")
        break


# In[15]:


#Extracting the URls
urls=driver.find_elements_by_xpath("//li[@class='EIR5N']/a")
for url in urls:
    URL.append(url.get_attribute('href'))


# In[16]:


print(len(URL))


# In[17]:


#Clicking back to "India"
click3=driver.find_element_by_xpath("//div[@class='_3zgcG']").click()


# In[18]:


#Clicking "Uttar Pradesh"
click4=driver.find_element_by_xpath("//ul[@class='_3eZBP']/li[3]").click()


# In[20]:


#Laoding data 
for i in range(0,50):
    try:
        click1=driver.find_element_by_xpath("//button[@class='rui-39-wj rui-3evoE rui-1JPTg']").click()
        time.sleep(5)
    except  NoSuchElementException:
        print("Cannot load more")
        break


# In[21]:


#Extracting the URls
urls=driver.find_elements_by_xpath("//li[@class='EIR5N']/a")
for url in urls:
    URL.append(url.get_attribute('href'))


# In[22]:


print(len(URL))


# In[23]:


#Clicking back to 'India'
click3=driver.find_element_by_xpath("//div[@class='_3zgcG']").click()


# In[24]:


#Clicking 'Kerela'
click4=driver.find_element_by_xpath("//ul[@class='_3eZBP']/li[4]").click()


# In[28]:


#Laoding data 
for i in range(0,50):
    try:
        click1=driver.find_element_by_xpath("//button[@class='rui-39-wj rui-3evoE rui-1JPTg']").click()
        time.sleep(5)
    except  NoSuchElementException:
        print("Cannot load more")
        break


# In[29]:


#Extracting the URls
urls=driver.find_elements_by_xpath("//li[@class='EIR5N']/a")
for url in urls:
    URL.append(url.get_attribute('href'))


# In[30]:


print(len(URL))


# In[31]:


#CLicking back to 'India'
click3=driver.find_element_by_xpath("//a[@class='_396bV _2zs4P']").click()


# In[32]:


#Clicking "Karnataka"
click4=driver.find_element_by_xpath("//ul[@class='_3eZBP']/li[5]").click()


# In[33]:


#Laoding data 
for i in range(0,50):
    try:
        click1=driver.find_element_by_xpath("//button[@class='rui-39-wj rui-3evoE rui-1JPTg']").click()
        time.sleep(5)
    except  NoSuchElementException:
        print("Cannot load more")
        break


# In[34]:


#Extracting the URls
urls=driver.find_elements_by_xpath("//li[@class='EIR5N']/a")
for url in urls:
    URL.append(url.get_attribute('href'))


# In[35]:


print(len(URL))


# In[36]:


#Clicking back to 'India'
click3=driver.find_element_by_xpath("//a[@class='_396bV _2zs4P']").click()   


# In[37]:


#Clicking 'Delhi'
click4=driver.find_element_by_xpath("//ul[@class='_3eZBP']/div/li[6]/a").click()


# In[38]:


#Laoding data 
for i in range(0,50):
    try:
        click1=driver.find_element_by_xpath("//button[@class='rui-39-wj rui-3evoE rui-1JPTg']").click()
        time.sleep(5)
    except  NoSuchElementException:
        print("Cannot load more")
        break


# In[39]:


#Extracting the URls
urls=driver.find_elements_by_xpath("//li[@class='EIR5N']/a")
for url in urls:
    URL.append(url.get_attribute('href'))


# In[40]:


print(len(URL))


# Creating List to store data

# In[42]:


BRAND=[]
MODEL=[]
VARI=[]
YR=[]
FUEL=[]
TRANS=[]
KM=[]
NOO=[]
PRICE=[]
LOC=[]


# In[43]:


# Extracting data from all the URls.
for url in URL[0:5360]:
    try:
        driver.get(url)
        time.sleep(2)
    except TimeoutException:
        print('TimeOut Exception')
    try:
        brand=driver.find_element_by_xpath("//div[@class='_3JPEe']/div[1]/div/span[2]")
        BRAND.append(brand.text)
    except NoSuchElementException:
        BRAND.append('-')
    try:
        model=driver.find_element_by_xpath("//div[@class='_3JPEe']/div[2]/div/span[2]")
        MODEL.append(model.text)
    except NoSuchElementException:
        MODEL.append('-')
    try:
        vari=driver.find_element_by_xpath("//div[@class='_3JPEe']/div[3]/div/span[2]")
        VARI.append(vari.text)
    except NoSuchElementException:
        VARI.append('-')
    try:
        yr=driver.find_element_by_xpath("//div[@class='_3JPEe']/div[4]/div/span[2]")
        YR.append(yr.text)
    except NoSuchElementException:
        YR.append('-')
    try:
        fuel=driver.find_element_by_xpath("//div[@class='_3JPEe']/div[5]/div/span[2]")
        FUEL.append(fuel.text)
    except NoSuchElementException:
        FUEL.append('-')
    try:
        trans=driver.find_element_by_xpath("//div[@class='_3JPEe']/div[6]/div/span[2]")
        TRANS.append(trans.text)
    except NoSuchElementException:
        TRANS.append('-')
    try:
        km=driver.find_element_by_xpath("//div[@class='_3JPEe']/div[7]/div/span[2]")
        KM.append(km.text)
    except NoSuchElementException:
        KM.append('-')
    try:
        noo=driver.find_element_by_xpath("//div[@class='_3JPEe']/div[8]/div/span[2]")
        NOO.append(noo.text)
    except NoSuchElementException:
        NOO.append('-')
    try:
        loc=driver.find_element_by_xpath("//span[@class='_2FRXm']")
        LOC.append(loc.text)
    except NoSuchElementException:
        LOC.append('-')
    try:
        price=driver.find_element_by_xpath("//span[@class='_2xKfz']")
        PRICE.append(price.text)
    except NoSuchElementException:
        PRICE.append('-')    


# In[44]:


print(len(BRAND),len(MODEL),len(VARI),len(YR),len(FUEL),len(TRANS),len(KM),len(NOO),len(LOC),len(PRICE))


# All lists have equal length

# Creating Data frame to store the data

# In[45]:


import pandas as pd


# In[46]:


car=pd.DataFrame()


# In[48]:


car['BRAND']=BRAND[:]
car['MODEL']=MODEL[:]
car['VARIANT']=VARI[:]
car['YEAR']=YR[:]
car['FUEL']=FUEL[:]
car['TRANSMISSION']=TRANS[:]
car['KM_Travelled']=KM[:]
car['No._of_owners']=NOO[:]
car['LOCATION']=LOC[:]
car['PRICE']=PRICE[:]


# In[49]:


car.head()


# In[50]:


#Shuffling the data in the data frame
car_shuffled=car.sample(frac=1).reset_index(drop=True)


# In[51]:


car_shuffled.to_csv('cars.csv')

