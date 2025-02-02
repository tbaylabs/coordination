const { createApp } = Vue;

createApp({
    data() {
        return {
            gridApi: null,
            columnDefs: [
                { field: 'model', sortable: true, filter: true },
                { field: 'task_set', sortable: true, filter: true },
                { field: 'condition', sortable: true, filter: true },
                { 
                    field: 'top_prop',
                    sortable: true,
                    valueFormatter: params => (params.value * 100).toFixed(1) + '%'
                },
                { 
                    field: 'absolute_diff',
                    sortable: true,
                    valueFormatter: params => params.value ? (params.value * 100).toFixed(1) + '%' : '-'
                },
                { 
                    field: 'percent_diff',
                    sortable: true,
                    valueFormatter: params => params.value ? (params.value).toFixed(1) + '%' : '-'
                },
                { 
                    field: 'p_value',
                    sortable: true,
                    valueFormatter: params => params.value ? params.value.toFixed(3) : '-'
                }
            ],
            includeUnanswered: true
        }
    },
    mounted() {
        const gridDiv = document.querySelector('#myGrid');
        const gridOptions = {
            columnDefs: this.columnDefs,
            rowData: benchmarkData,
            defaultColDef: {
                resizable: true,
            }
        };
        
        // Create new grid instance
        this.gridApi = agGridCommunity.createGrid(gridDiv, gridOptions);
        this.gridApi = gridOptions.api;
    },
    methods: {
        filterTaskSet(taskSet) {
            if (!this.gridApi) return;
            
            const filteredData = benchmarkData.filter(row => 
                row.task_set === taskSet && 
                row.unanswered_included === this.includeUnanswered
            );
            this.gridApi.setRowData(filteredData);
        }
    },
    watch: {
        includeUnanswered(newVal) {
            const currentTaskSet = this.gridApi.getFilterModel()?.task_set?.filter;
            if (currentTaskSet) {
                this.filterTaskSet(currentTaskSet);
            }
        }
    }
}).mount('#app');
