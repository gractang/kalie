import sys; sys.path.append('./')

BLACK_PIECE = "blue circle"
WHITE_PIECE = "white circle"
EMPTY_CELL = ' '

TTT_PLAYER_ONE = 'X'
TTT_PLAYER_TWO = 'O'
TTT_BOARD = "square game board"
TTT_PIECE_MAPPINGS = {WHITE_PIECE: TTT_PLAYER_ONE, BLACK_PIECE: TTT_PLAYER_TWO}
TTT_LANG_DESC = "ttt"
TTT_GAME_OVER = "Game Over"
TTT_ROBOT_PIECE = "blue piece"

OBJ_NAME_TO_VILD_NAME = {TTT_ROBOT_PIECE: BLACK_PIECE, "white piece": WHITE_PIECE}

VILD_MODEL_NAME = "vild" 
VILD_MODEL_PATH = "board_game_bot/object_detection/image_path_v2"

GRDINO_MODEL_NAME = "grounding dino"