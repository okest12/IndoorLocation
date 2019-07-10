import numpy as np
import xlwt
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
    with open('train.csv', 'r') as f:
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



def buildmap():
    linenum = 0
    global eNbcount
    global g_cellMap
    getmaxmap()
    with open('train.csv', 'r') as f:
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
    global maxX
    global maxY
    defaultpower = 9999.0
    numtoselect = 16
    numtoselect2 = 32
    buildmap()
    
    with open('test.csv', 'r') as f:
        linenum = 0
        
        file = xlwt.Workbook()
        defaule_style = xlwt.XFStyle()
        nearest_style = xlwt.easyxf('pattern: pattern solid, fore_colour blue')
        nearest_style2 = xlwt.easyxf('pattern: pattern solid, fore_colour red')
        new_style = xlwt.easyxf('pattern: pattern solid, fore_colour ice_blue')
        for line in f.readlines():
            if linenum != 0:
                tmpline = line.strip()
                posdata = tmpline.split(",")
                tmp= np.zeros([maxX+1,maxY+1])
                power_total = 0
                sheet_name = 'sheet name'+str(linenum)
                table = file.add_sheet(sheet_name)
                
                for i in range(0,eNbcount):

                    power_total += abs(float(posdata[i+2]))
                    tmp = tmp+np.abs(g_cellMap[i]-np.ones([maxX+1,maxY+1])*float(posdata[i+2]))
                    if i == eNbcount-1:
                        m = np.array(tmp)
                        min_powers = np.argpartition(m.ravel(),numtoselect)[:numtoselect]
                        min_powers2 = np.argpartition(m.ravel(),numtoselect2)[:numtoselect2]

                        #if linenum == 1:#this line is for shorter test time, to be delete
                        for k in range(0,maxX):
                            for n in range(0,maxY):
                                if m[k][n] == power_total:
                                    m[k][n]= defaultpower
                                    table.write(k,n,str(m[k][n]),new_style)
                                elif m[k][n] <= m.ravel()[min_powers[numtoselect-1]]:
                                    table.write(k,n,str(m[k][n]),nearest_style)
                                elif m[k][n] <= m.ravel()[min_powers2[numtoselect2-1]]:
                                    table.write(k,n,str(m[k][n]),nearest_style2)
                                else:
                                    table.write(k,n,str(m[k][n]),defaule_style)

                        '''
                        for j in range(0,numtoselect):
                            print(np.unravel_index(min_powers[j], m.shape))
                            print(m.ravel()[min_powers[j]])
                        print("\n")
                        f1.write("linenum"+str(linenum)+"\n")
                        for k in range(0,maxX):
                            for n in range(0,maxY):
                                if m[k][n] == power_total:
                                    m[k][n]= defaultpower
                                f1.write(str(m[k][n])+",")
                            f1.write("\n")
                        f1.write("\n")
                        '''
            linenum+=1
        file.save("reportsheets.xls")
    return
         
