import json


# API takes an argument and returns data in json format
def API(str=None):
    # Creates list for data
    coor = []
    radius = []
    col = []
    circ = []
    his = []
    acc = []
    for droplet in drops:
        coor.append(droplet.center)
        radius.append(droplet.radius)
        col.append(droplet.color)
        circ.append(droplet.circularity)
        his.append(droplet.history)
        acc.append(droplet.acc)
    if str is None or str == "all":
        jayson = json.dumps([drop.dump() for drop in drops])
        return jayson
    elif str == "coordinates":
        return json.dumps(coor)
    elif str == "radius":
        return json.dumps(radius)
    elif str == "color":
        return json.dumps(color)
    elif str == "circularity":
        return json.dumps(circ)
    elif str == "history":
        return json.dumps(his)
    elif str == "accuracy":
        return json.dumps(acc)
    else:
        return "Error: argument not valid"
