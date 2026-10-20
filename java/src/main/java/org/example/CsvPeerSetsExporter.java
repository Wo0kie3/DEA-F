package org.example;

import com.google.ortools.Loader;
import com.google.ortools.linearsolver.MPConstraint;
import com.google.ortools.linearsolver.MPSolver;
import com.google.ortools.linearsolver.MPVariable;
import tech.tablesaw.api.Table;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Maximal radial optimal reference set on a fixed CCR technology (no phase-II slacks). */
public class CsvPeerSetsExporter {
    private static final double PEER_TOL = 1e-7;

    public static void main(String[] args) throws Exception {
        if (args.length != 3) throw new IllegalArgumentException("reference.csv states.csv peers.csv");
        Loader.loadNativeLibraries();
        Table reference = Table.read().csv(args[0]);
        Table states = Table.read().csv(args[1]);
        List<String> inputs = columns(reference, "i");
        List<String> outputs = columns(reference, "o");
        if (inputs.isEmpty() || outputs.isEmpty() || reference.rowCount() == 0)
            throw new IllegalArgumentException("Nonempty inputs, outputs and reference required");
        validate(reference, inputs, outputs);
        validate(states, inputs, outputs);
        Path output = Path.of(args[2]);
        if (output.getParent() != null) Files.createDirectories(output.getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(output)) {
            writer.write("name,peer_refs,peer_count,peer_theta,peer_max_lambdas\n");
            for (int row = 0; row < states.rowCount(); row++) {
                MPSolver solver = MPSolver.createSolver("GLOP");
                if (solver == null) throw new IllegalStateException("GLOP unavailable");
                try {
                    double inf = MPSolver.infinity();
                    MPVariable theta = solver.makeNumVar(0, inf, "theta");
                    MPVariable[] lambda = solver.makeNumVarArray(reference.rowCount(), 0, inf);
                    for (String col : inputs) {
                        double scale = Math.max(reference.numberColumn(col).max(), states.numberColumn(col).getDouble(row));
                        MPConstraint constraint = solver.makeConstraint(-inf, 0);
                        constraint.setCoefficient(theta, -states.numberColumn(col).getDouble(row) / scale);
                        for (int j = 0; j < lambda.length; j++)
                            constraint.setCoefficient(lambda[j], reference.numberColumn(col).getDouble(j) / scale);
                    }
                    for (String col : outputs) {
                        double scale = Math.max(reference.numberColumn(col).max(), states.numberColumn(col).getDouble(row));
                        MPConstraint constraint = solver.makeConstraint(states.numberColumn(col).getDouble(row) / scale, inf);
                        for (int j = 0; j < lambda.length; j++)
                            constraint.setCoefficient(lambda[j], reference.numberColumn(col).getDouble(j) / scale);
                    }
                    solver.objective().setCoefficient(theta, 1);
                    solver.objective().setMinimization();
                    optimal(solver);
                    double optimum = theta.solutionValue();
                    // Fix the radial optimum, then explore alternative optima one lambda at a time.
                    theta.setBounds(optimum, optimum);
                    solver.objective().setCoefficient(theta, 0);
                    solver.objective().setMaximization();
                    List<String> peers = new ArrayList<>();
                    List<String> maxima = new ArrayList<>();
                    for (int j = 0; j < lambda.length; j++) {
                        solver.objective().setCoefficient(lambda[j], 1);
                        optimal(solver);
                        double maximum = lambda[j].solutionValue();
                        String name = reference.stringColumn("name").get(j);
                        maxima.add(name + "=" + maximum);
                        if (maximum > PEER_TOL) peers.add(name);
                        solver.objective().setCoefficient(lambda[j], 0);
                    }
                    if (peers.isEmpty()) throw new IllegalStateException("No positive peer coefficients");
                    writer.write(csv(states.stringColumn("name").get(row)) + "," + csv(String.join("|", peers))
                            + "," + peers.size() + "," + optimum + "," + csv(String.join("|", maxima)) + "\n");
                } finally {
                    solver.delete();
                }
            }
        }
        System.out.println("Peer sets exported for " + states.rowCount() + " states; lambda tolerance=" + PEER_TOL);
    }

    private static void optimal(MPSolver solver) {
        MPSolver.ResultStatus status = solver.solve();
        if (status != MPSolver.ResultStatus.OPTIMAL || !solver.verifySolution(1e-7, true))
            throw new IllegalStateException("Unverified/nonoptimal peer LP: " + status);
    }

    private static List<String> columns(Table table, String prefix) {
        return table.columnNames().stream().filter(c -> c.matches(prefix + "\\d+"))
                .sorted(Comparator.comparingInt(c -> Integer.parseInt(c.substring(1)))).toList();
    }

    private static void validate(Table table, List<String> inputs, List<String> outputs) {
        if (table.stringColumn("name").unique().size() != table.rowCount())
            throw new IllegalArgumentException("Duplicate state names");
        List<String> columns = new ArrayList<>(inputs);
        columns.addAll(outputs);
        for (int row = 0; row < table.rowCount(); row++) {
            String name = table.stringColumn("name").get(row);
            if (name == null || name.isBlank() || name.contains("|") || name.contains("="))
                throw new IllegalArgumentException("Invalid name for peer serialization");
            for (String col : columns) {
                double value = table.numberColumn(col).getDouble(row);
                if (!Double.isFinite(value) || value <= 0)
                    throw new IllegalArgumentException("This peer adapter requires strictly positive finite IO: " + col);
            }
        }
    }

    private static String csv(String value) {
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }
}
