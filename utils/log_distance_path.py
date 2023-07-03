def calc_log_distance_path(rss, a, gamma=3):
    '''
    calc_distance_log_path: 
        Parameter:
            - rss   : Mean RSSI Value
            - a     : Reference Power
            - gamma : Path Loss Exponent (def: 3)
    '''
    return pow(10, ((rss-a)/(-10*gamma)))