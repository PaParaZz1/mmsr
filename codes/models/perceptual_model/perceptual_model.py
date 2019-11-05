# Interface
class IPerceptualModel(object):
    def forward_feature(self, x, feature_name):
        # return list(Tensor)
        raise NotImplementedError
