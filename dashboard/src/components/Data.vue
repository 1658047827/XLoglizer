<template>
    <div>
        <el-table :data="tableData" style="width: 100%">
            <el-table-column prop="input" label="Input" width="350">
                <template #default="{ row }">
                    {{ showInput(row.input) }}
                </template>
            </el-table-column>
            <el-table-column prop="trace" label="Trace" width="180">
                <template #default="{ row }">
                    {{ showTrace(row.trace) }}
                </template>
            </el-table-column>
        </el-table>
        <el-pagination layout="prev, pager, next" :total="1000" />
    </div>
</template>

<script setup>

const showInput = (input) => {
    let formattedString = "";
    for (let i = 0; i < input.length - 1; i++) {
        formattedString += `E${input[i]}`;
        formattedString += ", ";
    }
    formattedString += `E${input[input.length - 1]}`;
    return formattedString;
}

const showTrace = (trace) => {
    let formattedString = "";
    for (let i = 0; i < trace.length - 1; i++) {
        formattedString += `S${trace[i]}`;
        formattedString += '→';
    }
    formattedString += `S${trace[trace.length - 1]}`;
    return formattedString;
}

const tableData = [
    {
        input: [22, 5, 5, 5, 26, 26, 26, 11, 9, 11],
        trace: [0, 1, 2, 3],
    },
]
</script>

<style scoped></style>