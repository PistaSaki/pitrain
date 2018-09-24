def concat_dics(*dics):
    dic = dict()
    for d in dics:
        if d is not None: 
            dic.update(d)
    return dic
    
if __name__ == "__main__":
    print( 
        'concat_dics({"a":1}, {1:"a"}) =', 
        concat_dics({"a":1}, {1:"a"})
    )
    