package org.example;

import put.dea.robustness.CCRExtremeEfficiencies;
import put.dea.robustness.ProblemData;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class CsvFrontierCandidateEvaluator {

    private static final double EFFICIENCY_TOL = 1e-9;

    public static void main(String[] args) throws IOException {
        String frontiersCsvPath = args.length > 0 ? args[0] : "../output/airports_frontiers.csv";
        String candidatesCsvPath = args.length > 1 ? args[1] : "../output/dea_search_space_RZE_to_front3.csv";
        String outputCsvPath = args.length > 2 ? args[2] : "../output/dea_search_space_RZE_to_front3_results.csv";
        int targetFrontier = args.length > 3 ? Integer.parseInt(args[3]) : 3;

        new CsvFrontierCandidateEvaluator().run(
                frontiersCsvPath,
                candidatesCsvPath,
                outputCsvPath,
                targetFrontier
        );

        System.out.println("Done. Results saved to: " + outputCsvPath);
    }

    public void run(
            String frontiersCsvPath,
            String candidatesCsvPath,
            String outputCsvPath,
            int targetFrontier
    ) throws IOException {

        List<DmuRow> allFrontierRows = readRowsFromCsv(frontiersCsvPath);
        List<DmuRow> candidates = readRowsFromCsv(candidatesCsvPath);

        List<String> inputNames = detectColumns(allFrontierRows.get(0).rawValues().keySet(), "i");
        List<String> outputNames = detectColumns(allFrontierRows.get(0).rawValues().keySet(), "o");

        List<DmuRow> referenceFrontier = allFrontierRows.stream()
                .filter(r -> {
                    String val = r.rawValues().get("frontier_layer");
                    return val != null && Integer.parseInt(val) == targetFrontier;
                })
                .collect(Collectors.toList());

        if (referenceFrontier.isEmpty()) {
            throw new IllegalArgumentException("No DMUs found for frontier_layer=" + targetFrontier);
        }

        System.out.println("==================================================");
        System.out.println("REFERENCE FRONTIER: " + targetFrontier);
        System.out.println("Number of DMUs in frontier: " + referenceFrontier.size());

        for (DmuRow r : referenceFrontier) {
            System.out.println("  - " + r.name());
        }

        System.out.println("==================================================");
        System.out.println("CANDIDATES FILE: " + candidatesCsvPath);
        System.out.println("TOTAL CANDIDATES: " + candidates.size());
        System.out.println("==================================================");

        List<String> referenceNames = referenceFrontier.stream()
                .map(DmuRow::name)
                .toList();

        CCRExtremeEfficiencies extremeEfficiencies = new CCRExtremeEfficiencies();

        Path outputPath = Path.of(outputCsvPath);
        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }

        try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
            writeHeader(writer, candidates.get(0).rawValues().keySet(), referenceNames);

            long startTime = System.currentTimeMillis();
            int total = candidates.size();
            int lastPct = -1;

            for (int i = 0; i < total; i++) {
                int pct = (int) ((i * 100.0) / total);

                if (pct != lastPct && pct % 5 == 0) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    double perItem = elapsed / (double) (i + 1);
                    long remaining = (long) (perItem * (total - i));

                    System.out.printf(
                            "Progress: %d%% (%d/%d) | ETA: %.1f sec%n",
                            pct,
                            i,
                            total,
                            remaining / 1000.0
                    );

                    lastPct = pct;
                }

                DmuRow candidate = candidates.get(i);

                List<DmuRow> evaluationSet = new ArrayList<>(referenceFrontier);
                evaluationSet.add(candidate);

                ProblemDataBundle bundle = buildProblemData(evaluationSet, inputNames, outputNames);
                List<Double> efficiencies = extremeEfficiencies.maxEfficiencyForAll(bundle.problemData());

                int candidateIdx = bundle.dmuNames().indexOf(candidate.name());
                if (candidateIdx < 0) {
                    throw new IllegalStateException("Candidate not found in evaluation set: " + candidate.name());
                }

                double candidateEfficiency = efficiencies.get(candidateIdx);
                boolean candidateEfficient = isEfficient(candidateEfficiency);

                Map<String, Double> refEfficiencies = new LinkedHashMap<>();
                for (String refName : referenceNames) {
                    int refIdx = bundle.dmuNames().indexOf(refName);
                    if (refIdx < 0) {
                        throw new IllegalStateException("Reference DMU not found in evaluation set: " + refName);
                    }
                    refEfficiencies.put(refName, efficiencies.get(refIdx));
                }

                writeRow(
                        writer,
                        candidate,
                        candidateEfficiency,
                        candidateEfficient,
                        refEfficiencies,
                        referenceNames
                );
            }

            System.out.printf("Progress: 100%% (%d/%d) | ETA: 0.0 sec%n", total, total);
        }
    }

    private void writeHeader(
            BufferedWriter writer,
            Set<String> candidateColumns,
            List<String> referenceNames
    ) throws IOException {
        List<String> cols = new ArrayList<>(candidateColumns);

        cols.add("candidate_efficiency");
        cols.add("candidate_efficient");

        for (String ref : referenceNames) {
            cols.add(ref + "_efficiency");
        }

        writer.write(String.join(",", cols));
        writer.newLine();
    }

    private void writeRow(
            BufferedWriter writer,
            DmuRow candidate,
            double candidateEfficiency,
            boolean candidateEfficient,
            Map<String, Double> refEfficiencies,
            List<String> referenceNames
    ) throws IOException {
        List<String> cols = new ArrayList<>();

        for (String col : candidate.rawValues().keySet()) {
            cols.add(escapeCsv(candidate.rawValues().get(col)));
        }

        cols.add(formatDouble(candidateEfficiency));
        cols.add(Boolean.toString(candidateEfficient));

        for (String ref : referenceNames) {
            cols.add(formatDouble(refEfficiencies.get(ref)));
        }

        writer.write(String.join(",", cols));
        writer.newLine();
    }

    private ProblemDataBundle buildProblemData(
            List<DmuRow> rows,
            List<String> inputNames,
            List<String> outputNames
    ) {
        double[][] inputs = new double[rows.size()][inputNames.size()];
        double[][] outputs = new double[rows.size()][outputNames.size()];
        List<String> dmuNames = new ArrayList<>();

        for (int r = 0; r < rows.size(); r++) {
            DmuRow row = rows.get(r);
            dmuNames.add(row.name());

            for (int c = 0; c < inputNames.size(); c++) {
                inputs[r][c] = parseDouble(row.rawValues().get(inputNames.get(c)), row.name(), inputNames.get(c));
            }

            for (int c = 0; c < outputNames.size(); c++) {
                outputs[r][c] = parseDouble(row.rawValues().get(outputNames.get(c)), row.name(), outputNames.get(c));
            }
        }

        ProblemData data = new ProblemData(inputs, outputs, inputNames, outputNames);
        return new ProblemDataBundle(data, dmuNames);
    }

    private List<DmuRow> readRowsFromCsv(String csvPath) throws IOException {
        List<DmuRow> rows = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(Path.of(csvPath))) {
            String headerLine = reader.readLine();
            if (headerLine == null || headerLine.isBlank()) {
                throw new IllegalArgumentException("CSV is empty: " + csvPath);
            }

            String[] headers = splitCsvLine(headerLine);

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }

                String[] parts = splitCsvLine(line);
                if (parts.length != headers.length) {
                    throw new IllegalArgumentException(
                            "Invalid CSV row. Expected " + headers.length + " columns but got " + parts.length +
                                    ". Row: " + line
                    );
                }

                Map<String, String> rawValues = new LinkedHashMap<>();
                for (int i = 0; i < headers.length; i++) {
                    rawValues.put(headers[i].trim(), unquote(parts[i].trim()));
                }

                String name = rawValues.get("name");
                if (name == null || name.isBlank()) {
                    throw new IllegalArgumentException("Encountered row with empty 'name'.");
                }

                rows.add(new DmuRow(name, rawValues));
            }
        }

        return rows;
    }

    private List<String> detectColumns(Set<String> allColumns, String prefix) {
        return allColumns.stream()
                .filter(col -> col.matches("^" + prefix + "\\d+$"))
                .sorted(Comparator.comparingInt(col -> Integer.parseInt(col.substring(1))))
                .collect(Collectors.toList());
    }

    private boolean isEfficient(double efficiency) {
        return Math.abs(efficiency - 1.0) <= EFFICIENCY_TOL || efficiency > 1.0;
    }

    private double parseDouble(String value, String dmuName, String colName) {
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid numeric value for DMU='" + dmuName + "', column='" + colName + "': " + value, e
            );
        }
    }

    private String formatDouble(double value) {
        return String.format(Locale.US, "%.10f", value);
    }

    private String escapeCsv(String value) {
        if (value == null) {
            return "";
        }
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    private String unquote(String value) {
        String v = value;
        if (v.startsWith("\"") && v.endsWith("\"") && v.length() >= 2) {
            v = v.substring(1, v.length() - 1).replace("\"\"", "\"");
        }
        return v;
    }

    private String[] splitCsvLine(String line) {
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;

        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);

            if (ch == '"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (ch == ',' && !inQuotes) {
                tokens.add(current.toString());
                current.setLength(0);
            } else {
                current.append(ch);
            }
        }

        tokens.add(current.toString());
        return tokens.toArray(new String[0]);
    }

    private record DmuRow(String name, Map<String, String> rawValues) {}
    private record ProblemDataBundle(ProblemData problemData, List<String> dmuNames) {}
}