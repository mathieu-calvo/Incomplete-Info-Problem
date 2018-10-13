
def action_input(question, answer=None):
    """
    Getting actions from players by prompting them to choose among a set of
    options

    Args:
        question (str): question to ask the player
        answer (obj): list of available choices, default None

    Returns:
        (str): action taken by player
    """
    answer = [i.lower() for i in answer]

    prompt = '/'.join(["\x1b[1;1m{0}\x1b[1;m".format(i) for i in answer])

    while True:
        choice = input("{0} [{1}]: ".format(question, prompt)).lower()

        if choice in answer:
            return choice

        valid_answer = {i[:len(choice)]: i for i in answer}

        if len(valid_answer) < len(answer):
            print(" -- Ambiguous, please use a more detailed answer.")
        elif choice in valid_answer:
            return valid_answer[choice]
        else:
            print(" -- Please answer only with {0} or {1}.".format
                  (", ".join(answer[:-1]), answer[-1]))


def amount_input(question, minimum=None, maximum=None, pot_size=None):
    """
    Getting amount to bet/raise from players by prompting them

    Args:
        question (str): question to ask the player
        minimum (int): minimum amount that is required, default None
        maximum (int): maximum amount that is required, default None
        pot_size (int): amount already in pot, default None

    Returns:
        (str): amount player wishes to bet/raise
    """
    if not (minimum is None) and (maximum is None):
        prompt = "min:{} - max:{} - pot_size:{}"\
            .format(minimum, maximum, pot_size)
    elif minimum:
        prompt = "min:{} - pot_size:{}".format(minimum, pot_size)
    else:
        prompt = "pot_size:{}".format(pot_size)

    while True:
        choice = int(input("{0} [{1}]: ".format(question, prompt)).lower())

        if not (minimum is None) and (maximum is None):
            if minimum <= choice <= maximum:
                return choice
            elif minimum <= choice:
                print(" -- Bet is too low")
            else:
                print(" -- Bet is too high")
        elif minimum:
            if choice >= minimum:
                return choice
            else:
                print(" -- Bet is too low")
        else:
            return choice
