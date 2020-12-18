import configparser
config = configparser.ConfigParser()
config.read('config.ini')
def getVariables():
    print(int(config['DEFAULT']['droplet_radius_min']))