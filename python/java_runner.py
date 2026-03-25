import subprocess


def _q(x):
    x = str(x)
    return f'"{x}"' if (" " in x or "/" in x or "\\" in x) else x


def _run_java_maven(java_entry, maven_executable, main_class, args_list):
    exec_args = " ".join(_q(a) for a in args_list)

    cmd = (
        f'{maven_executable} '
        f'compile exec:java '
        f'-Dexec.mainClass={main_class} '
        f'"-Dexec.args={exec_args}"'
    )

    print("Running command:")
    print(cmd)

    result = subprocess.run(cmd, cwd=java_entry, shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"Java failed for main class: {main_class}")


def generate_frontiers_with_java(
    input_csv,
    output_csv,
    java_entry,
    main_class,
    maven_executable="mvn",
):
    _run_java_maven(
        java_entry=java_entry,
        maven_executable=maven_executable,
        main_class=main_class,
        args_list=[input_csv, output_csv],
    )


def evaluate_candidates_with_java(
    frontiers_csv,
    candidates_csv,
    results_csv,
    target_front,
    java_entry,
    main_class,
    maven_executable="mvn",
):
    _run_java_maven(
        java_entry=java_entry,
        maven_executable=maven_executable,
        main_class=main_class,
        args_list=[frontiers_csv, candidates_csv, results_csv, target_front],
    )