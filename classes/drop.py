# Drop object class
class Drop:
    def __init__(self, center, radius, color, circularity):
        self.center = center
        self.radius = radius
        self.color = color
        self.circularity = circularity
        self.history = []
        self.acc = 0

    def __repr__(self):
        return ("( center = " + str(self.center) + ", radius = " + str(self.radius) + ", average color = "+str(self.color)+", circularity = "+str(round(self.circularity,3))+", length of history = "+str(len(self.history))+", accuracy = "+str((self.acc))+" ) ")

    def __str__(self):
        return ("( center = " + str(self.center) + ", radius = " + str(self.radius) + ", average color = "+str(self.color)+", circularity = "+str(round(self.circularity,3))+", length of history = "+str(len(self.history))+", accuracy = "+str((self.acc))+" ) ")

    def dump(self):
        return ("( center = " + str(self.center) + ", radius = " + str(self.radius) + ", average color = "+str(self.color)+", circularity = "+str(round(self.circularity,3))+", length of history = "+str(len(self.history))+", accuracy = "+str((self.acc))+" ) ")
