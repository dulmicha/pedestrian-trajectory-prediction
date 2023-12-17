class DistanceConverter:
    @staticmethod
    def pixels2meters(x, y):
        X_px2meters = (
            lambda x, y: (1 - ((1.47 * y + 161.76 - x) / (1.47 * y + 161.76))) * 8
        )
        Y_px2meters = (
            lambda y: 6.16793058e-07 * y**3
            - 8.61522438e-04 * y**2
            + 4.31688489e-01 * y
            - 4.75010213e01
        )
        return X_px2meters(x, y), Y_px2meters(y)

    @staticmethod
    def meters2pixels(x, y):
        X_meters2px = lambda x, y: (1.47 * y + 161.76) * x / 8
        Y_meters2px = (
            lambda y: 1.51690315e-02 * y**3
            - 3.20503299e-01 * y**2
            + 7.27107405e00 * y
            + 1.47945378e02
        )
        y_px = Y_meters2px(y)
        return X_meters2px(x, y_px), y_px
