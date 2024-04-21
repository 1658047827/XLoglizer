<template>
    <div style="font-size: x-large; margin-bottom: 10px; color: #409EFF;">Train Dataset</div>
    <div class="table-container">
        <el-table :data="tableData" style="width: 100%;" max-height="300px" height="300px">
            <el-table-column prop="index" label="Index" width="130"></el-table-column>
            <el-table-column prop="session_id" label="Session ID" width="250"></el-table-column>
            <el-table-column prop="input" label="Input" width="350">
                <template #default="{ row }">
                    {{ showInput(row.input) }}
                </template>
            </el-table-column>
            <el-table-column prop="trace" label="Trace" width="450">
                <template #default="{ row }">
                    {{ showTrace(row.trace) }}
                </template>
            </el-table-column>
            <el-table-column label="RNN Loglizer Topk Prediction">
                <template #default="{ row }">
                    <el-tag v-for="(val, index) in row.topk_value" :key="index" type="primary"
                        style="margin-right: 25px;">E{{ row.topk_pred[index] }}: {{ formatter(val) }}</el-tag>
                </template>
            </el-table-column>
            <el-table-column prop="label" label="Label" width="120">
                <template #default="{ row }">
                    E{{ row.label }}
                </template>
            </el-table-column>
        </el-table>
    </div>
    <div class="pagination">
        <el-pagination background layout="prev, pager, next" :page-size="6" :total="total"
            @current-change="pageChange" />
    </div>
</template>

<script setup>
import axios from "axios";
import { onMounted, ref } from "vue";

const total = ref(71007)

const pageChange = async (page) => {
    const params = {
        "page": page,
        "size": 6,
    };
    const resp = await axios.get("http://localhost:5000/dataset", { "params": params });
    tableData.value = resp.data.data;
    total.value = resp.data.total;
}

const topk = ref([0, 1, 2, 3, 4, 5, 6, 7, 8])

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

const formatter = (num) => {
    return num.toFixed(7)
}

const tableData = ref([
    {
        index: 0,
        session_id: "blk_3987090666992442841",
        input: [22, 5, 5, 5, 26, 26, 26, 11, 9, 11],
        trace: [0, 9, 12, 37, 4, 27, 23, 23, 12, 20, 20],
        topk_pred: [9, 11, 6, 5, 26, 21, 25, 3, 2],
        topk_value: [9.9904221e-01, 8.6151942e-04, 6.1206592e-05, 1.6576269e-05, 1.4534071e-05, 2.4041065e-06, 1.1216592e-06, 4.0975195e-07, 7.5171862e-08],
        label: 9,
    },
    {
        index: 1,
        session_id: "blk_3987090666992442841",
        input: [22, 5, 5, 5, 26, 26, 26, 11, 9, 11],
        trace: [0, 9, 12, 37, 4, 27, 23, 23, 12, 20, 20],
        topk_pred: [9, 11, 6, 5, 26, 21, 25, 3, 2],
        topk_value: [9.9904221e-01, 8.6151942e-04, 6.1206592e-05, 1.6576269e-05, 1.4534071e-05, 2.4041065e-06, 1.1216592e-06, 4.0975195e-07, 7.5171862e-08],
        label: 9
    },
    {
        index: 2,
        session_id: "blk_3987090666992442841",
        input: [22, 5, 5, 5, 26, 26, 26, 11, 9, 11],
        trace: [0, 9, 12, 37, 4, 27, 23, 23, 12, 20, 20],
        topk_pred: [9, 11, 6, 5, 26, 21, 25, 3, 2],
        topk_value: [9.9904221e-01, 8.6151942e-04, 6.1206592e-05, 1.6576269e-05, 1.4534071e-05, 2.4041065e-06, 1.1216592e-06, 4.0975195e-07, 7.5171862e-08],
        label: 9
    },
    {
        index: 3,
        session_id: "blk_3987090666992442841",
        input: [22, 5, 5, 5, 26, 26, 26, 11, 9, 11],
        trace: [0, 9, 12, 37, 4, 27, 23, 23, 12, 20, 20],
        topk_pred: [9, 11, 6, 5, 26, 21, 25, 3, 2],
        topk_value: [9.9904221e-01, 8.6151942e-04, 6.1206592e-05, 1.6576269e-05, 1.4534071e-05, 2.4041065e-06, 1.1216592e-06, 4.0975195e-07, 7.5171862e-08],
        label: 9
    },
    {
        index: 4,
        session_id: "blk_3987090666992442841",
        input: [22, 5, 5, 5, 26, 26, 26, 11, 9, 11],
        trace: [0, 9, 12, 37, 4, 27, 23, 23, 12, 20, 20],
        topk_pred: [9, 11, 6, 5, 26, 21, 25, 3, 2],
        topk_value: [9.9904221e-01, 8.6151942e-04, 6.1206592e-05, 1.6576269e-05, 1.4534071e-05, 2.4041065e-06, 1.1216592e-06, 4.0975195e-07, 7.5171862e-08],
        label: 9
    },
    {
        index: 5,
        session_id: "blk_3987090666992442841",
        input: [22, 5, 5, 5, 26, 26, 26, 11, 9, 11],
        trace: [0, 9, 12, 37, 4, 27, 23, 23, 12, 20, 20],
        topk_pred: [9, 11, 6, 5, 26, 21, 25, 3, 2],
        topk_value: [9.9904221e-01, 8.6151942e-04, 6.1206592e-05, 1.6576269e-05, 1.4534071e-05, 2.4041065e-06, 1.1216592e-06, 4.0975195e-07, 7.5171862e-08],
        label: 9
    },
])

onMounted(async () => {
    await pageChange(1);
})
</script>

<style scoped>
.table-container {
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 10px;
}

.pagination {
    margin-bottom: 10px;
    display: flex;
    justify-content: flex-end;
}
</style>