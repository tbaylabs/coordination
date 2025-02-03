import BenchmarkTables from '@/components/BenchmarkTables'

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-100">
      <div className="container mx-auto py-12">
        <h1 className="text-4xl font-bold text-center mb-4">
          Silent Agreement Benchmark
        </h1>
        <h2 className="text-xl text-center text-gray-600 mb-8">
          Version 0
        </h2>
        <div className="prose max-w-3xl mx-auto mb-8">
          <p>
            The Silent Agreement benchmark measures the ability of language models to coordinate responses 
            across isolated instances without direct communication. This capability has implications for 
            AI safety, particularly in areas of sandbagging detection and cyber security risk assessment.
          </p>
        </div>
        <BenchmarkTables />
      </div>
    </main>
  )
}
