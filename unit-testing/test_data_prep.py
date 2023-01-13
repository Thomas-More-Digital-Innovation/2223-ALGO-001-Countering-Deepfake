import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_prep as dp

def test_define_box_color_output():
    input_value = [0.3, 0.5, 0.8, 0.1]
    output_value = dp.define_box_color(input_value)
    assert output_value == [(0, 0, 255), (0, 255, 0), (0, 255, 0), (0, 0, 255)]
