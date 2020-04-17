$(document).ready(function () {
	var str = "sep=,\n";
	$("#saveCSVbtn").click(function() {
		$(".multiply_bet_history_table_row").children().not(".balance_before_after").each(function (i){
			$(this).children().each(function (j){
					if (j >= 7)
					{
						if (j == 7)
						{
							if ($(this).text().charCodeAt(0) == 10006)
								str = str + 'X';
							else
								str = str + ' ';
						}
					}
					else
					{
						str = str + $(this).text().replace(/(\r\n|\n|\r)/gm,"") + ",";
					}
			});
			str = str + '\n';
		});
		var blob = new Blob([str],
			{
				type: "application/json;utf - 8"
			}
		);
		var user_link = document.createElement('a');
		user_link.setAttribute('download', 'data.csv');
		user_link.setAttribute('href', window.URL.createObjectURL(blob));
		user_link.click();
	});
});