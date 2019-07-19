from inputs import get_gamepad


btn_dic = {'BTN_WEST': 'X', 'BTN_NORTH': 'Y', 'BTN_EAST': 'B', 'BTN_SOUTH': 'A'}

digital_dic = {'BTN_TRIGGER_HAPPY1': 'left', 'BTN_TRIGGER_HAPPY2': 'right',
              'BTN_TRIGGER_HAPPY3': 'forward', 'BTN_TRIGGER_HAPPY4': 'backward',
              'BTN_SOUTH': 'shoot'}

digital_idx_dic = {'forward': 0, 'backward': 1,  'left': 2, 'right': 3,
                   'shoot': 4}

N = 5 #correspond to five possible actions

def test_btn_ctrl():
    while 1:
        events = get_gamepad()
        for event in events:
            if event.code != 'SYN' and event.state == 1:
                    print(btn_dic[event.code])

def get_triggered_button(last_action=[0]*N):

    e = last_action[:]
    events = get_gamepad()
    for event in events:
        if event.code in digital_dic.keys():
            e[digital_idx_dic[digital_dic[event.code]]] = event.state
    return e


