desc = ["FFFFFFFHFFFFF",
        "FFFFPPPHPPPFF",
        "FFHFPFPHPFGFF",
        "FFHFPFPPPFFFF",
        "FFHHPFFHFFFFF",
        "FFFFSFFHFFFFF",
        "FFFFFFFHFFFFF"]

num_rows = len(desc)
num_cols = len(desc[0])
row = 5
col = 4
state = desc[row][col]
push_right_states = []
push_up_states = []
push_left_states = []
push_down_states = []
last_act = None

######## ADD SHAPING REWARDS TO THE PATH ########
while state != "G":
    #check left
    if last_act != 2 and col > 0:
        if desc[row][col-1] == "P" or desc[row][col-1] == "G":
            push_left_states.append(row*num_cols+col)
            col = col-1
            state = desc[row][col]

            last_act = 0
            continue
    #check down
    if last_act != 3 and row < num_rows-1:
        if desc[row+1][col] == "P" or desc[row+1][col] == "G":
            push_down_states.append(row*num_cols+col)
            row = row+1
            state = desc[row][col]

            last_act = 1
            continue

    #check right
    if last_act != 0 and col < num_cols-1:
        if desc[row][col+1] == "P" or desc[row][col+1] == "G":
            push_right_states.append(row*num_cols+col)
            
            col = col+1
            state = desc[row][col]

            last_act = 2
            continue

    #check up
    if last_act != 1 and row > 0:
        if desc[row-1][col] == "P" or desc[row-1][col] == "G":
            push_up_states.append(row*num_cols+col)
            
            row = row-1
            state = desc[row][col]

            last_act = 3
            continue

    print(row,col, last_act)

for row in range(num_rows):
    for col in range(num_cols):
        if col > 0 and desc[row][col-1] == "P" and desc[row][col] == "F":
            push_left_states.append(row*num_cols+col)
        if col < num_cols-1 and desc[row][col+1] == "P" and desc[row][col] == "F":
            push_right_states.append(row*num_cols+col)
        if row > 0 and desc[row-1][col] == "P" and desc[row][col] == "F":
            push_up_states.append(row*num_cols+col)
        if row < num_rows-1 and desc[row+1][col] == "P" and desc[row][col] == "F":
            push_down_states.append(row*num_cols+col)

print("push_right_states = ", push_right_states)
print("push_up_states = ", push_up_states)
print("push_down_states = ", push_down_states)
print("push_left_states = ", push_left_states)

print("--------------------")
hole_positions = []
for row in range(len(desc)):
    for col in range(len(desc[row])):
        if desc[row][col] == "H":
            hole_positions.append(row*num_cols+col)

print("hole_positions = ", hole_positions)