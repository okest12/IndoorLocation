import numpy as np
np.set_printoptions(threshold=np.inf)

g_cellMap=[]
maxX = 0
maxY = 0
eNbcount = 0

def fillmatrix(tmpline):
    global eNbcount
    if len(tmpline):
        posdata = tmpline.split(",")
        #print(posdata)
        x = int(posdata[0])
        y = int(posdata[1])
        for i in range(0,eNbcount):
            g_cellMap[i][x][y] = float(posdata[i+2])
        
def getmaxmap():
    global maxX
    global maxY
    isfirstline = 1
    with open(r'C:\03-study\MachineLearning\train.csv', 'r') as f:
        for line in f.readlines():
            if isfirstline:
                isfirstline=0
            else:
                pos = line.strip().split(",",3)
                x = int(pos[0])
                y = int(pos[1])
                if x>maxX:
                   maxX= x
                if y>maxY:
                   maxY= y
        #print(maxX," ",maxY)
                
#def buildinitmaps():


def buildmap():
    linenum = 0
    global eNbcount
    global g_cellMap
    getmaxmap()
    with open(r'C:\03-study\MachineLearning\train.csv', 'r') as f:
        for line in f.readlines():
            if linenum == 0:
                firstline = line.strip()
                #print(firstline)
                eNbcount = firstline.count(",")-1
                #print(eNbcount)

                for i in range (0,eNbcount):
                    g_cellMap.append(np.zeros([maxX+1,maxY+1]))
                #print(g_cellMap[0])
                
            else:
                tmpline = line.strip()
                fillmatrix(tmpline)
                
            linenum +=1
        #print(g_cellMap[0][0])

def buildtestmap():
    global g_cellMap
    global eNbcount
    buildmap()
    
    with open(r'C:\03-study\MachineLearning\test.csv', 'r') as f:
        linenum = 0

        f1=open(r"C:\03-study\MachineLearning\report.csv","a")


        for line in f.readlines():
            if linenum != 0:
                tmpline = line.strip()
                posdata = tmpline.split(",")
                tmp= np.zeros([maxX+1,maxY+1])
                for i in range(0,eNbcount):
                    tmp = tmp+np.abs(g_cellMap[i]-np.ones([maxX+1,maxY+1])*float(posdata[i+2]))
                    if (i==eNbcount-1):
                        minrow = tmp.min()
                        m = np.array(tmp)
                        print(np.where(m == minrow))
                        print("min: %s",minrow)
                        
                        f1.write("linenum"+str(linenum)+"\n")
                        f1.write(str(tmp)+"\n")
            linenum+=1
    
            
    return
         
buildtestmap()