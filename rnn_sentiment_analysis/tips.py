
class Error_Tips(object):

    red_color = '\033[0;31m'
    color_back = '\033[0m'

    Value_Range = '[0.0 ~ 1.0]'
    ERROR_WORD_SEQUENCE_NOT_FIT = "Error : Word_Sequence no fit operation"
    ERROR_WORD_SEQUENCE_FIT_ONCE = "Error : Word_Sequence only fit once"
    ERROR_NOT_SUPPORT_GPU = "Error : device does not support GPU "

    def Error_File_not_found(self, file):
        result = Error_Tips.red_color + \
                 "Error: File not exists! File: {}".format(file) + \
                 Error_Tips.color_back
        return result

    def Error_Range_of_value(self, name, range):
        result = Error_Tips.red_color + \
                 "Error: {} Range of value is {}".format(name, range) + \
                 Error_Tips.color_back
        return result

    def Error_Length_Same(self, val_1, val_2):
        result = Error_Tips.red_color + \
                 "Error: {} and {} the Length must be the same.".format(val_1, val_2) + \
                 Error_Tips.color_back
        return result

    def Error_empty_value(self, val_1):
        result = Error_Tips.red_color + \
                 "{} Can't be empty".format(val_1) + \
                 Error_Tips.color_back
        return result

    def Error_empty_value_when(self, val_1, val_2, val_3):
        result = Error_Tips.red_color + \
                 "When using {},  cannot be empty, {} is {} ".format(val_1, val_2, val_3) + \
                 Error_Tips.color_back

        return result


def test():
    error = Error_Tips()
    print(error.Error_File_not_found('/home/file'))
    print(error.Error_Range_of_value('test', '[0 ~ 1]'))
    print(error.Error_Length_Same('value2', 'value2'))

if __name__ == '__main__':
    test()