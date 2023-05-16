LAKE_DESC = ["FFHFFFFFFFFHFFFFF",
             "FFHFFFFFFFFHPPPFF",
             "FFHFFFHFFFFHPFGFF",
             "FFHFFFHFFFPPPFFFF",
             "FFFSFFHHFFPHFFFFF",
             "FHFPFFFFFFPHFFFFF",
             "FFFPPPPPPPPHFFFFF"]

num_rows = len(LAKE_DESC)
num_cols = len(LAKE_DESC[0])

for row in range(num_rows):
    for col in range(num_cols):
        if LAKE_DESC[row][col] == "S":
            start_row = row
            start_col = col
            break

row = start_row
col = start_col
state = LAKE_DESC[row][col]
push_right_states = []
push_up_states = []
push_left_states = []
push_down_states = []
last_act = None

######## ADD SHAPING REWARDS TO THE PATH ########
while state != "G":
    #check left
    if last_act != 2 and col > 0:
        if LAKE_DESC[row][col-1] == "P" or LAKE_DESC[row][col-1] == "G":
            push_left_states.append(row*num_cols+col)
            col = col-1
            state = LAKE_DESC[row][col]

            last_act = 0
            continue
    #check down
    if last_act != 3 and row < num_rows-1:
        if LAKE_DESC[row+1][col] == "P" or LAKE_DESC[row+1][col] == "G":
            push_down_states.append(row*num_cols+col)
            row = row+1
            state = LAKE_DESC[row][col]

            last_act = 1
            continue

    #check right
    if last_act != 0 and col < num_cols-1:
        if LAKE_DESC[row][col+1] == "P" or LAKE_DESC[row][col+1] == "G":
            push_right_states.append(row*num_cols+col)
            
            col = col+1
            state = LAKE_DESC[row][col]

            last_act = 2
            continue

    #check up
    if last_act != 1 and row > 0:
        if LAKE_DESC[row-1][col] == "P" or LAKE_DESC[row-1][col] == "G":
            push_up_states.append(row*num_cols+col)
            
            row = row-1
            state = LAKE_DESC[row][col]

            last_act = 3
            continue

    print(row,col, last_act)

for row in range(num_rows):
    for col in range(num_cols):
        if col > 0 and LAKE_DESC[row][col-1] == "P" and LAKE_DESC[row][col] == "F":
            push_left_states.append(row*num_cols+col)
        if col < num_cols-1 and LAKE_DESC[row][col+1] == "P" and LAKE_DESC[row][col] == "F":
            push_right_states.append(row*num_cols+col)
        if row > 0 and LAKE_DESC[row-1][col] == "P" and LAKE_DESC[row][col] == "F":
            push_up_states.append(row*num_cols+col)
        if row < num_rows-1 and LAKE_DESC[row+1][col] == "P" and LAKE_DESC[row][col] == "F":
            push_down_states.append(row*num_cols+col)

print("push_right_states = ", push_right_states)
print("push_up_states = ", push_up_states)
print("push_down_states = ", push_down_states)
print("push_left_states = ", push_left_states)

print("--------------------")
hole_positions = []
for row in range(len(LAKE_DESC)):
    for col in range(len(LAKE_DESC[row])):
        if LAKE_DESC[row][col] == "H":
            hole_positions.append(row*num_cols+col)

print("hole_positions = ", hole_positions)