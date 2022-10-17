import os

def addHeaderToMetadata(filename, header):
    file = os.path.join(filename)

    f = open(file, 'a')

    f.write('\n--------------------- ' + header + ' ---------------------\n')

    f.close()

def addToNetMetadata(filename, key, value, header=''):
    file = os.path.join(filename)

    f = open(file, 'a')

    if header != '':
        f.write('\n--------------------- ' + header + ' ---------------------\n')

    f.write(key + ':' + ' ' + str(value) + '\n')

    f.close()
