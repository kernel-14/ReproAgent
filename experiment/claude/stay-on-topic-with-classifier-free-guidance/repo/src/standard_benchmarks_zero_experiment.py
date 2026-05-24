from stay_on_topic_cfg.experiments import load_default_matrix


def main():
    matrix = load_default_matrix()
    for task in matrix.zero_shot_tasks:
        print(task)


if __name__ == "__main__":
    main()

