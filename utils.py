import numpy as np
import datetime
import glob

def read_data(dfolder = '/Users/nespinoza/github/coronavirus/data', sdate_string = '2020-03-03', region = 'Metropolitana'):
    if not dfolder[-1] == '/':
        dfolder = dfolder+'/'
    files = glob.glob(dfolder+'*.csv')
    sdate = datetime.datetime.strptime(sdate_string,'%Y-%m-%d')
    days = np.array([])
    infected = np.array([])
    for i in range(len(files)):
        # Get dates since sdate_string:
        date = datetime.datetime.strptime(files[i].split('/')[-1].split('.')[0].split('_')[-1],'%Y-%m-%d')
        days = np.append(days,(date - sdate).days)
        # Open CSV, get desired region, save:
        infected = np.append(infected, get_infected(region,files[i]))
    return days, infected

def get_infected(region, fname):
    f = open(fname,'r')
    while True:
        line = f.readline()
        if line != '':
            r,cn,ct,cr,cf,fecha = line.split(';')
            if r.lower() in region.lower():
                return np.double(ct)
        else:
            break
    return 0.
