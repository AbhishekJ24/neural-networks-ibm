import java.util.Scanner;

public class hill_climbing {

	
	public static void main(String[] args) {
		int n,i,j;
	
		Scanner sc=new Scanner(System.in);
		System.out.println("Enter number of nodes in graph");
		n=sc.nextInt();
		
			
		int[][] graph=new int[n][n];
		
		for(i=0;i<n;i++)
			for(j=0;j<n;j++)
				graph[i][j]=0;
						
		for(i=0;i<n;i++)
		{
			for(j=i+1;j<n;j++)
			{
				System.out.println("Is "+i+" is connected to "+ j);	
				graph[i][j]=sc.nextInt();
			}
		}
		System.out.println("The adjacency matrix is:");
		for(i=0;i<n;i++)
		{
			for(j=0;j<n;j++)
			{		System.out.print(graph[i][j]+ "\t");
			}
			System.out.println();
		}
		
		}

}
