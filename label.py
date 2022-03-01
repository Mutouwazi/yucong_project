def labeling(x,ch):
    # (12, 6, 3, 4, 735)
    # (12, 4, 3, 4, 316)

    if ch == 'V1':
        locnum = 6
    else:
        locnum = 4
    x4 = x % 4

    x = (x - x4)/4
    x3 = x % 3

    x = (x-x3)/3
    x2 = x % locnum

    x1 = (x-x2)/locnum

    if ch == 'V1':
        if x3 == 2:
            return 'd',x4
        elif x3 == 1 and x2 == 1:
            return 'c',x4
        elif x3 == 0 and x2 == 3:
            return 'a',x4
        elif x3 == 0 and x2 > 3:
            return 'e',x4
        elif x3 == 0 and x2 < 3:
            return 'b',x4
        else:
            return 'z',x4
    else:
        #return x1, x2, x3, x4
        if x3 == 2:
            return 'd', x4
        elif x3 == 1 and x2 == 1:
            return 'c', x4
        elif x3 == 0 and x2 == 1:
            return 'a', x4
        elif x3 == 0 and x2 > 1:
            return 'e', x4
        elif x3 == 0 and x2 < 1:
            return 'b', x4
        else:
            return 'z', x4
