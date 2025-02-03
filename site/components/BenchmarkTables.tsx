import { Table, TableContainer, TableHeader, TableCell, TableBody } from '@/components/ui/table'
import { useState, useEffect } from 'react'

const BenchmarkTables = () => {
  const [currentView, setCurrentView] = useState<'COT' | 'PURE'>('COT')
  const [data, setData] = useState<any[]>([])
  
  useEffect(() => {
    fetchData()
  }, [])
  
  // Fetch and process benchmark data
  const fetchData = async () => {
    try {
      const response = await fetch('/data/benchmark_table.csv')
      const csv = await response.text()
      const rows = csv.split('\n').slice(1).map(row => {
        const [model, top_prop, top_prop_sem, top_prop_ci_lower_95] = row.split(',')
        return {
          model,
          top_prop: parseFloat(top_prop),
          top_prop_sem: parseFloat(top_prop_sem),
          top_prop_ci_lower_95: parseFloat(top_prop_ci_lower_95)
        }
      })
      setData(rows)
    } catch (error) {
      console.error('Error fetching benchmark data:', error)
    }
  }

  return (
    <div className="flex flex-col space-y-4">
      <div className="flex gap-2">
        <button
          onClick={() => setCurrentView('COT')}
          className={`px-4 py-2 rounded ${
            currentView === 'COT' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-200 text-gray-700'
          }`}
        >
          COT View
        </button>
        <button
          onClick={() => setCurrentView('PURE')}
          className={`px-4 py-2 rounded ${
            currentView === 'PURE'
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-200 text-gray-700'
          }`}
        >
          PURE View
        </button>
      </div>
      
      <TableContainer>
        <Table>
          <TableHeader>
            <TableCell>Model</TableCell>
            <TableCell>Silent Agreement Score</TableCell>
          </TableHeader>
          <TableBody>
            {data.map((row, index) => (
              <tr key={index}>
                <TableCell>{row.model}</TableCell>
                <TableCell 
                  className={`${
                    row.top_prop_ci_lower_95 > 0 
                      ? 'text-green-600' 
                      : 'text-red-600'
                  }`}
                >
                  {row.top_prop_ci_lower_95?.toFixed(2)}
                </TableCell>
              </tr>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </div>
  )
}

export default BenchmarkTables
