import subprocess


def export_possible_goals_with_java(reference_csv, states_csv, top_targets_csv, output_csv,
                                    java_entry, maven_executable="mvn"):
    _run_java_maven(java_entry, maven_executable, "org.example.CsvPossibleGoalsExporter",
                    [reference_csv, states_csv, top_targets_csv, output_csv])


def export_peer_sets_with_java(reference_csv, states_csv, output_csv, java_entry,
                               maven_executable="mvn"):
    _run_java_maven(java_entry, maven_executable, "org.example.CsvPeerSetsExporter",
                    [reference_csv, states_csv, output_csv])


def _q(x):
    x = str(x)
    # Maven parses the inner quotes; the shell receives one double-quoted property.
    if "'" in x or '"' in x:
        raise ValueError("Maven arguments containing quote characters are unsupported")
    return f"'{x}'" if (any(c.isspace() for c in x) or "/" in x or "\\" in x) else x


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


def generate_preference_relations_with_java(
    input_csv,
    output_csv,
    java_entry,
    main_class,
    frontier_layer=None,
    maven_executable="mvn",
):
    args_list = [input_csv, output_csv]
    if frontier_layer is not None:
        args_list.append(frontier_layer)

    _run_java_maven(
        java_entry=java_entry,
        maven_executable=maven_executable,
        main_class=main_class,
        args_list=args_list,
    )


def evaluate_pairwise_possible_candidates_with_java(
    reference_csv,
    candidates_csv,
    results_csv,
    java_entry,
    main_class,
    maven_executable="mvn",
):
    _run_java_maven(
        java_entry=java_entry,
        maven_executable=maven_executable,
        main_class=main_class,
        args_list=[reference_csv, candidates_csv, results_csv],
    )


def export_extreme_ranks_with_java(
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


def export_extreme_efficiencies_with_java(
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


def export_candidate_robust_metrics_with_java(
    reference_csv,
    candidates_csv,
    output_csv,
    java_entry,
    main_class,
    maven_executable="mvn",
):
    _run_java_maven(
        java_entry=java_entry,
        maven_executable=maven_executable,
        main_class=main_class,
        args_list=[reference_csv, candidates_csv, output_csv],
    )
