# Auto-anchor utilities stub for compatibility

def check_anchor_order(m):
    """Check anchor order - anchors should be in ascending order"""
    pass

def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """Check anchors against dataset"""
    pass

def kmean_anchors(path='data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, dir_name='.'):
    """K-means anchor analysis"""
    pass
