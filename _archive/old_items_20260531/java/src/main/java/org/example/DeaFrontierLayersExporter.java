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

public class DeaFrontierLayersExporter {

    private static final double EFFICIENCY_TOL = 1e-9;

    public static void main(String[] args) throws IOException {
        String inputCsvPath = args.length > 0 ? args[0] : "data/airports.csv";
        String outputCsvPath = args.length > 1 ? args[1] : "output/airports_frontiers.csv";

        new DeaFrontierLayersExporter().run(inputCsvPath, outputCsvPath);

        System.out.println("Done. Frontier layers saved to: " + outputCsvPath);
    }

    public void run(String inputCsvPath, String outputCsvPath) throws IOException {
        List<DmuRow> allRows = readDmuRowsFromCsv(inputCsvPath);

        if (allRows.isEmpty()) {
            throw new IllegalArgumentException("Input CSV contains no rows.");
        }

        List<String> inputNames = detectColumns(allRows.get(0).rawValues().keySet(), "i");
        List<String> outputNames = detectColumns(allRows.get(0).rawValues().keySet(), "o");

        if (inputNames.isEmpty()) {
            throw new IllegalArgumentException("No input columns found. Expected columns like i1,i2,...");
        }
        if (outputNames.isEmpty()) {
            throw new IllegalArgumentException("No output columns found. Expected columns like o1,o2,...");
        }

        List<DmuRow> remaining = new ArrayList<>(allRows);
        List<ResultRow> finalResults = new ArrayList<>();

        int currentLayer = 1;
        CCRExtremeEfficiencies extremeEfficiencies = new CCRExtremeEfficiencies();

        while (!remaining.isEmpty()) {
            ProblemDataBundle bundle = buildProblemData(remaining, inputNames, outputNames);

            List<Double> efficiencies = extremeEfficiencies.maxEfficiencyForAll(bundle.problemData());

            List<DmuRow> efficientRows = new ArrayList<>();
            List<DmuRow> inefficientRows = new ArrayList<>();

            for (int i = 0; i < remaining.size(); i++) {
                DmuRow row = remaining.get(i);
                double efficiency = efficiencies.get(i);
                boolean isEfficient = isEfficient(efficiency);

                ResultRow result = new ResultRow(
                        row.name(),
                        new LinkedHashMap<>(row.rawValues()),
                        efficiency,
                        isEfficient,
                        currentLayer
                );
                finalResults.add(result);

                if (isEfficient) {
                    efficientRows.add(row);
                } else {
                    inefficientRows.add(row);
                }
            }

            if (efficientRows.isEmpty()) {
                throw new IllegalStateException(
                        "No efficient DMUs found on layer " + currentLayer +
                        ". Check tolerance or DEA solver behavior."
                );
            }

            System.out.println("Layer " + currentLayer + ": " + efficientRows.size() + " efficient DMUs");

            remaining = inefficientRows;
            currentLayer++;
        }

        writeResults(outputCsvPath, allRows, finalResults);
    }

    private boolean isEfficient(double efficiency) {
        return Math.abs(efficiency - 1.0) <= EFFICIENCY_TOL || efficiency > 1.0;
    }

    private ProblemDataBundle buildProblemData(
            List<DmuRow> rows,
            List<String> inputNames,
            List<String> outputNames
    ) {
        double[][] inputs = new double[rows.size()][inputNames.size()];
        double[][] outputs = new double[rows.size()][outputNames.size()];
        List<String> names = new ArrayList<>();

        for (int r = 0; r < rows.size(); r++) {
            DmuRow row = rows.get(r);
            names.add(row.name());

            for (int c = 0; c < inputNames.size(); c++) {
                inputs[r][c] = parseDouble(row.rawValues().get(inputNames.get(c)), row.name(), inputNames.get(c));
            }

            for (int c = 0; c < outputNames.size(); c++) {
                outputs[r][c] = parseDouble(row.rawValues().get(outputNames.get(c)), row.name(), outputNames.get(c));
            }
        }

        ProblemData data = new ProblemData(inputs, outputs, inputNames, outputNames);
        return new ProblemDataBundle(data, names);
    }

    private List<DmuRow> readDmuRowsFromCsv(String csvPath) throws IOException {
        List<DmuRow> rows = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(Path.of(csvPath))) {
            String headerLine = reader.readLine();
            if (headerLine == null || headerLine.isBlank()) {
                throw new IllegalArgumentException("CSV is empty: " + csvPath);
            }

            String[] headers = splitCsvLine(headerLine);
            List<String> headerList = Arrays.stream(headers)
                    .map(String::trim)
                    .collect(Collectors.toList());

            if (!headerList.contains("name")) {
                throw new IllegalArgumentException("CSV must contain 'name' column.");
            }

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

    private void writeResults(
            String outputCsvPath,
            List<DmuRow> originalRows,
            List<ResultRow> finalResults
    ) throws IOException {
        Path outputPath = Path.of(outputCsvPath);
        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }

        Map<String, ResultRow> resultMap = new LinkedHashMap<>();
        for (ResultRow row : finalResults) {
            resultMap.put(row.name(), row);
        }

        List<String> originalColumns = new ArrayList<>(originalRows.get(0).rawValues().keySet());

        try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
            List<String> header = new ArrayList<>(originalColumns);
            header.add("efficiency_on_layer");
            header.add("efficient_on_layer");
            header.add("frontier_layer");

            writer.write(String.join(",", header));
            writer.newLine();

            for (DmuRow original : originalRows) {
                ResultRow result = resultMap.get(original.name());
                if (result == null) {
                    throw new IllegalStateException("Missing result for DMU: " + original.name());
                }

                List<String> cols = new ArrayList<>();
                for (String col : originalColumns) {
                    cols.add(escapeCsv(original.rawValues().get(col)));
                }

                cols.add(formatDouble(result.efficiencyOnLayer()));
                cols.add(Boolean.toString(result.efficientOnLayer()));
                cols.add(Integer.toString(result.frontierLayer()));

                writer.write(String.join(",", cols));
                writer.newLine();
            }
        }
    }

    private List<String> detectColumns(Set<String> allColumns, String prefix) {
        return allColumns.stream()
                .filter(col -> col.matches("^" + prefix + "\\d+$"))
                .sorted(Comparator.comparingInt(col -> Integer.parseInt(col.substring(1))))
                .collect(Collectors.toList());
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

    /**
     * Minimal CSV split supporting quoted commas.
     */
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

    private record DmuRow(
            String name,
            Map<String, String> rawValues
    ) {}

    private record ProblemDataBundle(
            ProblemData problemData,
            List<String> dmuNames
    ) {}

    private record ResultRow(
            String name,
            Map<String, String> rawValues,
            double efficiencyOnLayer,
            boolean efficientOnLayer,
            int frontierLayer
    ) {}
}