def ssd_loss(y_true, y_pred):
    loc_loss = localization_loss(y_true, y_pred)
    clf_loss = classfication_loss(y_true, y_pred)
    return clf_loss + loc_loss