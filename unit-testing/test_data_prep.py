# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import data_prep as dp

def define_box_color(list_of_predictions):
    box_color_list = []
    red = (0, 0, 255)
    green = (0, 255, 0)
    for i in list_of_predictions:
        if i < 0.5:
            box_color_list.append(red)
        else:
            box_color_list.append(green)
    return box_color_list

def test_define_box_color_output():
    input_value = [0.3, 0.5, 0.8, 0.1]
    output_value = define_box_color(input_value)
    assert output_value == [(0, 0, 255), (0, 255, 0), (0, 255, 0), (0, 0, 255)]
