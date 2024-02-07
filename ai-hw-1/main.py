from utils.utils import *
import argparse
import path_finding as pf

def main():
    parser = argparse.ArgumentParser(
        prog="COMSW4701 HW1",
        description="Robot Path Planning",
    )
    parser.add_argument(
        "world_path", help="The directory containing worlds saved as .npy files"
    )
    parser.add_argument(
        "world_id", help="The world that we are testing on"
    )
    parser.add_argument(
        "-e", help="A* options (1 for Manhattan, 2 for Euclidean, 3 for Manhattan and IDA*, 4 for Euclidean and IDA*)"
    )
    parser.add_argument(
        "-b", help="Beam width for beam search (default 100)"
    )
    parser.add_argument(
        "-a", help="Animation options (1 for expanded nodes, 2 for path)"
    )
    args = parser.parse_args()

    id = int(args.world_id)
    if args.e is not None and 1 < int(args.e) < 4:
        h = int(args.e)
    else:
        h = 0
    if args.b is not None and int(args.b) > 0:
        width = int(args.b)
    else:
        width = 100
    if args.a is not None and 0 < int(args.a) < 3:
        a = int(args.a)
    else:
        a = 0

    print("=" * 40)
    print("Testing Grid World Path Planning...")
    print(f"Loading grid world file from path: {args.world_path}")
    if h == 0:
        print("Modes: 1. DFS, 2. BFS")
    elif h == 1:
        print("Modes: 1. A_STAR, 2. BEAM_A_STAR")
        print("Using Manhattan heuristic")
    elif h == 2:
        print("Modes: 1. A_STAR, 2. BEAM_A_STAR")
        print("Using Euclidean heuristic")
    elif h == 3:
        print("Mode: ID_A_STAR")
        print("Using Manhattan heuristic")
    elif h == 4:
        print("Mode: ID_A_STAR")
        print("Using Euclidean heuristic")

    start_goal = [
        [(10, 10), (87, 87)],
        [(10, 10), (90, 90)],
        [(10, 10), (90, 90)],
        [(24, 24), (43, 42)],
    ]

    if 0 < id < 5:
        start, goal = start_goal[id - 1]
    else:
        start, goal = [(0, 0), (0, 0)]
    pf.test_world(
        id, start, goal, h, width, a, args.world_path
    )

    print("Done")
    print("=" * 40)

if __name__ == "__main__":
    main()
    