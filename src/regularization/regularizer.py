# -*- coding: utf-8 -*-
__author__ = 'maoss2'


class _Regularizer(object):
    """
    Parent class of Regularizers
    """

    def __init__(self, model, name=None):
        super(_Regularizer, self).__init__()
        self.name = name
        self.model = model

    def loss_regularized(self, reg_loss_function):
        raise NotImplementedError

    def loss_all_params_regularized(self, reg_loss_function):
        raise NotImplementedError


class L1Regularizer(_Regularizer):
    """
    L1 regularized loss
    """
    def __init__(self, model, lambda_reg, name=None):
        super(L1Regularizer, self).__init__(name=name, model=model)
        self.lambda_reg = lambda_reg

    def loss_regularized(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if self.name == model_param_name:
                assert self.name is None, "the name can't be None"
                reg_loss_function += self.lambda_reg * L1Regularizer.__add_l1(var=model_param_value)

        return reg_loss_function

    def loss_all_params_regularized(self, reg_loss_function):
        return self.lambda_reg * L1Regularizer.__add_l1_all(reg_loss_function, self.model)

    @staticmethod
    def __add_l1(var):
        return var.abs().sum()

    @staticmethod
    def __add_l1_all(reg_loss_function, model):
        for param_name, param in model.named_parameters():
            reg_loss_function += L1Regularizer.__add_l1(param)
        return reg_loss_function


class L2Regularizer(_Regularizer):
    """
       L2 regularized loss
       """

    def __init__(self, model, lambda_reg, name=None):
        super(L2Regularizer, self).__init__(name=name, model=model)
        self.lambda_reg = lambda_reg

    def loss_regularized(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            assert self.name is None, "the name can't be None"
            if self.name == model_param_name:
                reg_loss_function += self.lambda_reg * L2Regularizer.__add_l2(var=model_param_value)

        return reg_loss_function

    def loss_all_params_regularized(self, reg_loss_function):
        return self.lambda_reg * L2Regularizer.__add_l2_all(reg_loss_function, self.model)

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum()

    @staticmethod
    def __add_l2_all(reg_loss_function, model):
        for param_name, param in model.named_parameters():
            reg_loss_function += L2Regularizer.__add_l2(param)
        return reg_loss_function


class ElasticNetRegularizer(_Regularizer):
    """
    Elastic Net Regularizer
    """
    def __init__(self, model, lambda_reg, alpha_reg, name=None):
        super(ElasticNetRegularizer).__init__(model=model, name=name)
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg

    def loss_regularized(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            assert self.name is None, "the name can't be None"
            if self.name == model_param_name:
                reg_loss_function += self.lambda_reg * \
                                     (((1 - self.alpha_reg) * ElasticNetRegularizer.__add_l2(var=model_param_value)) +
                                      (self.alpha_reg * ElasticNetRegularizer.__add_l1(var=model_param_value)))
                # reg_loss_function += (1 - self.alpha_reg) * ElasticNetRegularizer.__add_l2(var=model_param_value)
                # reg_loss_function += self.alpha_reg * ElasticNetRegularizer.__add_l1(var=model_param_value)
                # reg_loss_function += self.lambda_reg * reg_loss_function
        return reg_loss_function

    def loss_all_params_regularized(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            reg_loss_function += self.lambda_reg * \
                                 (((1 - self.alpha_reg) * ElasticNetRegularizer.__add_l2(var=model_param_value)) +
                                  (self.alpha_reg * ElasticNetRegularizer.__add_l1(var=model_param_value)))
        return reg_loss_function

    @staticmethod
    def __add_l1(var):
        return var.abs().sum()

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum()


class GroupSparseLassoRegularizer(_Regularizer):
    """
    Group Sparse Lasso Regularizer
    """
    def __init__(self, model, lambda_reg, alpha_reg, name=None):
        super(GroupSparseLassoRegularizer).__init__(model=model, name=name)
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg
        self.reg_l2_l1 = ElasticNetRegularizer(model=self.model, lambda_reg=self.lambda_reg,
                                               alpha_reg=self.alpha_reg, name=self.name)
        self.reg_l1 = L1Regularizer(model=self.model, lambda_reg=self.lambda_reg, name=self.name)

    def loss_regularized(self, reg_loss_function):
        reg_loss_function = self.reg_l2_l1.loss_regularized(reg_loss_function=reg_loss_function) +\
                            self.reg_l1.loss_regularized(reg_loss_function=reg_loss_function)
        return reg_loss_function

    def loss_all_params_regularized(self, reg_loss_function):
        reg_loss_function = self.reg_l2_l1.loss_all_params_regularized(reg_loss_function=reg_loss_function) + \
                            self.reg_l1.loss_all_params_regularized(reg_loss_function=reg_loss_function)
        return reg_loss_function


class GroupLassoRegularizer(_Regularizer):
    """
    GroupLasso Regularizer
    """
    def __init__(self, model, name):
        super(GroupLassoRegularizer).__init__(model=model, name=name)

    @staticmethod
    def __grouplasso_reg(groups, dim):
        if dim == -1:
            # We only have single group
            return groups.norm(2)
        return groups.norm(2, dim=dim).sum()
